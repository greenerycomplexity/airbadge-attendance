#!/usr/bin/env python3
"""
AirBadge Attendance Processor
Hardware-accelerated via Apple Vision Neural Engine on Apple Silicon
"""

import sys
import re
from pathlib import Path
from datetime import date, timedelta, datetime
from dataclasses import dataclass
from zoneinfo import ZoneInfo

ROME = ZoneInfo("Europe/Rome")

def _today() -> date:
    """Current date in Rome time (CET/CEST) — always used for future-day cutoff."""
    return datetime.now(ROME).date()


# ── Auto-install dependencies ─────────────────────────────────────────────────

def _ensure_deps():
    import importlib
    import subprocess

    # (import_name, pip_package)
    required = [
        ("Foundation", "pyobjc"),   # umbrella; brings Vision + Foundation
        ("PIL",        "Pillow"),
        ("numpy",      "numpy"),
        ("rich",       "rich"),
    ]

    missing_pkgs = []
    for import_name, pip_name in required:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing_pkgs.append(pip_name)

    if missing_pkgs:
        print(f"Installing missing packages: {', '.join(missing_pkgs)} …")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", *missing_pkgs],
            check=False,
        )
        if result.returncode != 0:
            print("pip install failed. Try manually:")
            print(f"  pip3 install {' '.join(missing_pkgs)}")
            sys.exit(1)
        print("Done. Starting…\n")

_ensure_deps()

# Safe to import now — packages are guaranteed to be present
from Foundation import NSURL
from Vision import VNRecognizeTextRequest, VNImageRequestHandler
from PIL import Image
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

# ── Constants ─────────────────────────────────────────────────────────────────

MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}
MONTH_NAMES = [""] + list(MONTHS.keys())
DAYS = {"Mon", "Tue", "Wed", "Thu", "Fri"}

DAY_MINUTES = 4 * 60   # 4 h per working day

# Vision OCR sometimes reads the digit '0' as the letter 'o' in low-height
# bar values (e.g. "0.00" → "o.00"). Normalise before parsing.
_OCR_DIGIT_FIXES = str.maketrans("oO", "00")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DayData:
    date: date
    minutes_done: int
    is_holiday: bool


@dataclass
class WeekData:
    days: list
    source_file: str

    @property
    def monday(self) -> date:
        return self.days[0].date

    @property
    def friday(self) -> date:
        return self.days[-1].date

    @property
    def label(self) -> str:
        mon, fri = self.days[0].date, self.days[-1].date
        if mon.month == fri.month:
            return f"{mon.day}–{fri.day} {MONTH_NAMES[mon.month]} {mon.year}"
        return (f"{mon.day} {MONTH_NAMES[mon.month]}"
                f" – {fri.day} {MONTH_NAMES[fri.month]} {fri.year}")

    @property
    def total_done(self) -> int:
        return sum(d.minutes_done for d in self.days)

    @property
    def working_days(self) -> int:
        return sum(1 for d in self.days if not d.is_holiday)

    @property
    def pending_days(self) -> list:
        """Non-holiday days that are today or in the future (Rome time) — excluded from target."""
        today = _today()
        return [d for d in self.days if not d.is_holiday and d.date >= today]

    @property
    def is_in_progress(self) -> bool:
        """True if the week contains today or future dates."""
        today = _today()
        return any(d.date >= today for d in self.days)

    @property
    def target(self) -> int:
        """Expected hours: only counts non-holiday days strictly before today (Rome time).
        Today and future dates are excluded — their hours may not be complete yet."""
        today = _today()
        past_working = sum(1 for d in self.days if not d.is_holiday and d.date < today)
        return past_working * DAY_MINUTES

    @property
    def missing(self) -> int:
        return max(0, self.target - self.total_done)


# ── Helpers ───────────────────────────────────────────────────────────────────

def hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def parse_time(s: str) -> int:
    """'4:00' / '3:49' / '0.00' / 'o.00' → minutes"""
    s = s.strip().translate(_OCR_DIGIT_FIXES).replace(".", ":")
    if ":" not in s:
        return 0
    h, m = s.split(":", 1)
    try:
        return int(h) * 60 + int(m)
    except ValueError:
        return 0


# ── Apple Vision OCR (Neural Engine) ─────────────────────────────────────────

def ocr_image(path: Path) -> list[dict]:
    """
    Run Apple Vision text recognition on an image file.
    Returns observations with text and normalised bounding boxes.
    Vision coordinate system: origin bottom-left, y increases upward.
    """
    url = NSURL.fileURLWithPath_(str(path.resolve()))
    handler = VNImageRequestHandler.alloc().initWithURL_options_(url, None)

    req = VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(1)          # 1 = VNRequestTextRecognitionLevelAccurate
    req.setUsesLanguageCorrection_(False)
    req.setMinimumTextHeight_(0.008)

    ok, _err = handler.performRequests_error_([req], None)
    if not ok or req.results() is None:
        return []

    obs = []
    for result in req.results():
        cands = result.topCandidates_(1)
        if not cands or len(cands) == 0:
            continue
        text = cands[0].string()
        b = result.boundingBox()
        cx = b.origin.x + b.size.width / 2
        cy = b.origin.y + b.size.height / 2   # Vision y (0=bottom)
        obs.append({
            "text": text,
            "x": b.origin.x, "y": b.origin.y,
            "w": b.size.width, "h": b.size.height,
            "cx": cx, "cy": cy,
        })
    return obs


# ── Red-text detector (holiday labels) ───────────────────────────────────────

def is_red(img: Image.Image, o: dict) -> bool:
    """
    Sample the pixel region of observation `o` and decide if the text is
    rendered in red (holiday marker).  Converts Vision coords → PIL coords.
    """
    iw, ih = img.size
    x1 = max(0, int(o["x"] * iw))
    y1 = max(0, int((1 - o["y"] - o["h"]) * ih))   # flip y axis
    x2 = min(iw, int((o["x"] + o["w"]) * iw))
    y2 = min(ih, int((1 - o["y"]) * ih))

    if x2 <= x1 or y2 <= y1:
        return False

    px = np.array(img.crop((x1, y1, x2, y2)).convert("RGB"), dtype=np.float32)
    if px.size == 0:
        return False

    r, g, b = px[:, :, 0], px[:, :, 1], px[:, :, 2]
    non_bg  = (r < 240) | (g < 240) | (b < 240)    # exclude near-white
    red_px  = non_bg & (r > 160) & (g < 100) & (b < 100) & (r > g + 80) & (r > b + 80)

    total = int(np.sum(non_bg))
    return total > 0 and (int(np.sum(red_px)) / total) > 0.12


# ── Screenshot parser ─────────────────────────────────────────────────────────

_MONTH_RE   = re.compile(r"^(" + "|".join(MONTHS) + r")\s+(\d{4})$")
# Allow no-space variants ("Mon8", "Wed1") that Vision produces for tight labels
_DAY_RE     = re.compile(r"^(Mon|Tue|Wed|Thu|Fri)\s*(\d{1,2})$")
_DAYNAME_RE = re.compile(r"^(Mon|Tue|Wed|Thu|Fri)$")
# Vision sometimes reads '0' as 'o'/'O' for small zero-height bars (e.g. "o.00")
_TIME_RE    = re.compile(r"^[0-9oO]{1,2}[:.][0-9oO]{2}$")


def parse_screenshot(path: Path) -> "WeekData | None":
    raw = ocr_image(path)
    if not raw:
        console.print(f"[red]  OCR returned nothing for {path.name}[/red]")
        return None

    img = Image.open(path)

    month_year = None
    day_labels = []     # (cx, day_name, day_num, obs)
    time_cands = []     # (cx, cy, minutes, obs)
    day_name_obs = []   # isolated "Mon" / "Tue" etc. (split label fallback)
    num_obs = []        # isolated numbers that might pair with day names

    for o in raw:
        t = o["text"].strip()

        m = _MONTH_RE.match(t)
        if m:
            month_year = (MONTHS[m.group(1)], int(m.group(2)))
            continue

        m = _DAY_RE.match(t)
        if m:
            day_labels.append((o["cx"], m.group(1), int(m.group(2)), o))
            continue

        if _DAYNAME_RE.match(t):
            day_name_obs.append(o)
            continue

        if _TIME_RE.match(t):
            time_cands.append((o["cx"], o["cy"], parse_time(t), o))
            continue

        # bare day-number (may pair with a split day-name above)
        if re.match(r"^\d{1,2}$", t) and 1 <= int(t) <= 31:
            num_obs.append(o)

    # ── Fallback: stitch split "Mon" + "29" observations ─────────────────────
    if len(day_labels) < 5 and day_name_obs:
        for dn_obs in day_name_obs:
            # find the bare number observation closest in x and within ±3% y
            best, best_dx = None, 1.0
            for n_obs in num_obs:
                dy = abs(n_obs["cy"] - dn_obs["cy"])
                dx = abs(n_obs["cx"] - dn_obs["cx"])
                if dy < 0.03 and dx < best_dx:
                    best, best_dx = n_obs, dx
            if best:
                merged_cx = (dn_obs["cx"] + best["cx"]) / 2
                merged_obs = {
                    **dn_obs,
                    "x": min(dn_obs["x"], best["x"]),
                    "w": max(dn_obs["x"] + dn_obs["w"], best["x"] + best["w"])
                         - min(dn_obs["x"], best["x"]),
                    "cx": merged_cx,
                }
                day_labels.append((merged_cx, dn_obs["text"].strip(),
                                   int(best["text"].strip()), merged_obs))

    # ── Validate ──────────────────────────────────────────────────────────────
    if not month_year:
        console.print(f"[yellow]  ⚠ No month/year in {path.name}[/yellow]")
        return None

    day_labels.sort(key=lambda d: d[0])
    day_labels = day_labels[:5]   # keep leftmost 5

    if len(day_labels) != 5:
        console.print(
            f"[yellow]  ⚠ Found {len(day_labels)}/5 day labels in {path.name}[/yellow]")
        return None

    avg_day_cy = sum(d[3]["cy"] for d in day_labels) / 5

    # Chart time values sit ABOVE (higher Vision y) the day labels, but below
    # the phone status bar clock (cy ≈ 0.96).  "This week" totals sit below the
    # day labels (cy < avg_day_cy).
    CLOCK_CUTOFF = 0.90
    chart_times = [t for t in time_cands
                   if avg_day_cy < t[1] < CLOCK_CUTOFF]

    if len(chart_times) != 5:
        # Widen lower bound: zero-bar values can sit very close to day labels
        chart_times = [t for t in time_cands
                       if t[1] > avg_day_cy - 0.01 and t[1] < CLOCK_CUTOFF]
        chart_times.sort(key=lambda t: t[0])
        # Take 5 closest to day-label x positions
        if len(chart_times) > 5:
            day_xs = [d[0] for d in day_labels]
            chart_times = sorted(
                chart_times,
                key=lambda t: min(abs(t[0] - dx) for dx in day_xs)
            )[:5]

    if len(chart_times) != 5:
        console.print(
            f"[red]  ✗ Found {len(chart_times)}/5 chart values in {path.name}[/red]")
        return None

    chart_times.sort(key=lambda t: t[0])

    # ── Build WeekData ────────────────────────────────────────────────────────
    month_num, year = month_year
    mon_num = day_labels[0][2]
    try:
        monday = date(year, month_num, mon_num)
    except ValueError:
        console.print(f"[red]  ✗ Bad date {year}-{month_num}-{mon_num}[/red]")
        return None

    days = []
    for i in range(5):
        days.append(DayData(
            date=monday + timedelta(days=i),
            minutes_done=chart_times[i][2],
            is_holiday=is_red(img, day_labels[i][3]),
        ))

    return WeekData(days=days, source_file=path.name)


# ── Gap detection ─────────────────────────────────────────────────────────────

def find_gaps(weeks: list[WeekData]) -> list[tuple]:
    """Return list of (gap_monday, gap_friday, n_weeks) for interior gaps."""
    gaps = []
    for i in range(len(weeks) - 1):
        cur_fri   = weeks[i].friday
        nxt_mon   = weeks[i + 1].monday
        exp_mon   = cur_fri + timedelta(days=3)   # expected next Monday
        if nxt_mon > exp_mon:
            n = (nxt_mon - exp_mon).days // 7
            gaps.append((exp_mon, nxt_mon - timedelta(days=3), n))
    return gaps


# ── Chip detection ────────────────────────────────────────────────────────────

def _chip() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        return "Apple Silicon"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    if not folder.exists() or not folder.is_dir():
        console.print(f"[red]Folder not found:[/red] {folder}")
        sys.exit(1)

    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    images = sorted(p for p in folder.iterdir() if p.suffix in exts)

    if not images:
        console.print(f"[dim]No PNG/JPG screenshots found in[/dim] {folder.resolve()}")
        console.print("[dim]Nothing to process. Add screenshots and try again.[/dim]")
        sys.exit(0)

    console.print()
    console.print("[bold cyan]AirBadge Attendance Processor[/bold cyan]"
                  f"  [dim]· Apple Vision Neural Engine · {_chip()}[/dim]")
    console.print(f"[dim]Scanning {len(images)} screenshot(s)…[/dim]")
    console.print()

    weeks = []
    for p in images:
        console.print(f"[dim]  → {p.name}[/dim]", end="")
        w = parse_screenshot(p)
        if w:
            console.print(f"  [green]✓ {w.label}[/green]")
            weeks.append(w)
        else:
            console.print("  [red]✗ skipped[/red]")

    if not weeks:
        console.print("\n[red]No weeks parsed. Exiting.[/red]")
        sys.exit(1)

    # Sort and deduplicate
    weeks.sort(key=lambda w: w.monday)
    seen, unique = set(), []
    for w in weeks:
        if w.monday not in seen:
            seen.add(w.monday)
            unique.append(w)
    weeks = unique

    # ── Week breakdown ────────────────────────────────────────────────────────
    console.print()
    tbl = Table(
        title="[bold white]Week-by-Week Breakdown[/bold white]",
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold bright_white",
        min_width=60,
        show_lines=False,
    )
    tbl.add_column("Week",         style="white",      min_width=32)
    tbl.add_column("Time Done",    style="green",      justify="center", min_width=11)
    tbl.add_column("Time Missing", style="red",        justify="center", min_width=13)
    tbl.add_column("Working Days", style="dim white",  justify="center", min_width=13)

    grand_done = grand_miss = 0
    has_in_progress = False

    for w in weeks:
        grand_done += w.total_done
        grand_miss += w.missing

        if w.working_days == 0:
            tbl.add_row(
                f"[dim]{w.label}[/dim]",
                f"[dim]{hhmm(w.total_done)}[/dim]",
                "[dim]— holiday[/dim]",
                "[dim]0 / 5[/dim]",
            )
        elif w.is_in_progress:
            has_in_progress = True
            n_pending = len(w.pending_days)
            n_counted = w.working_days - n_pending
            done_style = "green" if w.missing == 0 else "yellow"
            tbl.add_row(
                f"{w.label}  [dim cyan]↻ in progress[/dim cyan]",
                f"[{done_style}]{hhmm(w.total_done)}[/{done_style}]",
                f"[red]{hhmm(w.missing)}[/red]  [dim](past days only)[/dim]"
                    if w.missing else "[dim cyan]—[/dim cyan]",
                f"[dim]{n_counted} counted · {n_pending} pending[/dim]",
            )
        else:
            done_style = "green" if w.missing == 0 else "yellow"
            tbl.add_row(
                w.label,
                f"[{done_style}]{hhmm(w.total_done)}[/{done_style}]",
                f"[red]{hhmm(w.missing)}[/red]" if w.missing else "[green]00:00[/green]",
                f"{w.working_days} / 5",
            )

    console.print(tbl)

    if has_in_progress:
        console.print(
            "[dim cyan]  ↻ in progress[/dim cyan]"
            "[dim]  —  today and future dates are not counted towards missing time"
            " (Rome time, CET/CEST)[/dim]"
        )

    # ── Totals ────────────────────────────────────────────────────────────────
    console.print()
    tot = Table(
        title="[bold white]Overall Totals[/bold white]",
        box=box.ROUNDED,
        border_style="bright_green",
        header_style="bold bright_white",
        min_width=44,
        show_lines=False,
    )
    tot.add_column("Metric",  style="white",  min_width=22)
    tot.add_column("Value",   justify="center", min_width=12)

    tot.add_row("Total Time Done",    f"[bold green]{hhmm(grand_done)}[/bold green]")
    tot.add_row("Total Time Missing", f"[bold red]{hhmm(grand_miss)}[/bold red]")

    console.print(tot)

    # ── Missing-weeks notice ──────────────────────────────────────────────────
    gaps = find_gaps(weeks)
    if gaps:
        lines = []
        for gap_mon, gap_fri, n in gaps:
            wk_label = (
                f"{gap_mon.day}–{gap_fri.day} {MONTH_NAMES[gap_mon.month]} {gap_mon.year}"
                if gap_mon.month == gap_fri.month
                else (f"{gap_mon.day} {MONTH_NAMES[gap_mon.month]}"
                      f" – {gap_fri.day} {MONTH_NAMES[gap_fri.month]} {gap_fri.year}")
            )
            plural = "week" if n == 1 else "weeks"
            lines.append(
                f"  [yellow]•[/yellow] {n} {plural}:  [yellow]{wk_label}[/yellow]"
            )

        console.print()
        console.print(Panel(
            "\n".join(lines),
            title="[bold yellow]⚠  Missing Time Periods Detected[/bold yellow]",
            subtitle="[dim]Please add screenshots for the weeks listed above[/dim]",
            border_style="yellow",
            padding=(1, 2),
        ))

    console.print()


if __name__ == "__main__":
    main()
