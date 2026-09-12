"""Appendix G - planned vs actual project schedule.

Regenerates `figures/gantt_planned_vs_actual.png`.

The original figure was produced ad hoc and only the PNG survived; the schedule
below was recovered from that image and is now the source of truth. Edit the
SCHEDULE table and re-run:

    python scripts/make_gantt.py

Weeks are inclusive of the start tick and exclusive of the end tick, i.e. a task
running 1 -> 3 occupies weeks 1 and 2 and is two weeks long.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT = Path(__file__).resolve().parent.parent
FIGURES = PROJECT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Palette shared with every other figure in the project (see notebooks 02 and 04).
BLUE, RUST = "#3B6EA5", "#B3512F"
INK, MUTED, FAINT = "#1A1A1A", "#6B6B6B", "#D8D8D8"

# task, planned (start, end), actual (start, end)
SCHEDULE = [
    ("Topic definition and proposal",        (1, 3),  (1, 4)),
    ("Literature review",                    (3, 6),  (3, 7)),
    ("Ethics classification and approval",   (4, 5),  (4, 5)),
    ("Data acquisition",                     (6, 7),  (6, 8)),
    ("Preprocessing pipeline (two tracks)",  (7, 8),  (7, 9)),
    ("Model implementation",                 (8, 10), (9, 11)),
    ("Hyper-parameter search and training",  (10, 12), (11, 14)),
    ("Evaluation, figures, result export",   (12, 13), (13, 14)),
    ("Streamlit application",                (13, 14), (13, 15)),
    ("User evaluation",                      (14, 15), (15, 16)),
    ("Report writing and revision",          (14, 17), (14, 17)),
]

LAST_WEEK = max(max(p[1], a[1]) for _, p, a in SCHEDULE)

plt.rcParams.update({
    "figure.dpi": 200,
    "font.size": 9.5,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": INK,
})

fig, ax = plt.subplots(figsize=(11.2, 6.2))

BAR_H = 0.34
OFFSET = 0.19

for row, (task, (p0, p1), (a0, a1)) in enumerate(SCHEDULE):
    y = row
    # Planned sits above actual, matching the legend's reading order.
    ax.barh(y - OFFSET, p1 - p0, left=p0, height=BAR_H,
            color=BLUE, zorder=3)
    ax.barh(y + OFFSET, a1 - a0, left=a0, height=BAR_H,
            color=RUST, zorder=3)

    # A thin connector makes a delayed *start* visible, not just a late finish.
    if a0 > p0:
        ax.plot([p0, a0], [y + OFFSET, y + OFFSET],
                color=RUST, lw=0.9, ls=(0, (2, 2)), zorder=2)

    slip = a1 - p1
    if slip > 0:
        ax.text(a1 + 0.22, y + OFFSET,
                f"+{slip} wk" if slip == 1 else f"+{slip} wks",
                va="center", ha="left", fontsize=8.5,
                color=RUST, fontweight="bold", zorder=4)
    else:
        ax.text(a1 + 0.22, y + OFFSET, "on time",
                va="center", ha="left", fontsize=8, color=MUTED,
                style="italic", zorder=4)

ax.set_yticks(range(len(SCHEDULE)))
ax.set_yticklabels([t for t, _, _ in SCHEDULE], fontsize=9.5)
ax.invert_yaxis()                       # week 1 at the top
ax.set_ylim(len(SCHEDULE) - 0.45, -0.55)

ax.set_xticks(range(1, LAST_WEEK + 1))
ax.set_xlim(0.55, LAST_WEEK + 2.3)      # headroom for the slip labels
ax.set_xlabel("Project week", fontsize=10, labelpad=8)

ax.xaxis.grid(True, color=FAINT, lw=0.7, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", length=3)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(FAINT)

# The bars cascade top-left to bottom-right, so the top-right corner is the one
# reliably empty region. Putting the legend there fixes the fault in the first
# version of this figure, where it sat on top of the last two bars.
ax.legend(
    handles=[Patch(facecolor=BLUE, label="Planned"),
             Patch(facecolor=RUST, label="Actual")],
    loc="upper right", bbox_to_anchor=(0.995, 0.995),
    frameon=False, fontsize=10, handlelength=1.6,
    handleheight=0.9, labelspacing=0.6, borderaxespad=0.0,
)

slipped = sum(1 for _, p, a in SCHEDULE if a[1] > p[1])
fig.suptitle("Project schedule: planned against actual",
             x=0.5, y=0.985, fontsize=14, fontweight="bold")
fig.text(0.5, 0.928,
         f"{slipped} of {len(SCHEDULE)} stages overran, most by a single week; "
         f"the week-{LAST_WEEK} completion date held",
         ha="center", va="top", fontsize=9.5, color=MUTED)

fig.tight_layout(rect=(0, 0, 1, 0.905))
out = FIGURES / "gantt_planned_vs_actual.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)

# The same table as text, for the appendix caption or the report body.
print(f"\n{'stage':38s} {'planned':>9s} {'actual':>9s} {'slip':>6s}")
for task, (p0, p1), (a0, a1) in SCHEDULE:
    print(f"{task:38s} {f'{p0}-{p1}':>9s} {f'{a0}-{a1}':>9s} {a1 - p1:>6d}")
