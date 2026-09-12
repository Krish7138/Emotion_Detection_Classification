"""Architecture diagrams for the report and the presentation.

Regenerates:
    figures/arch_highlevel.png   four layers and the flow between them
    figures/arch_design.png      the two-by-two experimental design

Both were previously produced ad hoc and only the PNGs survived. Two defects in
those versions are corrected here:

  * arch_highlevel had no arrow from the data layer into the processing layer,
    and the notebook-to-artefact arrows cut diagonally across the layer banner,
    so the diagram did not read as a single flow. Layer names now sit in a left
    column and every connector runs straight down an empty channel.
  * arch_design printed the delta F1 label on top of the two result boxes.

Figure numbering lives in the caption of whatever document uses these,
not inside the image.

Run:  python scripts/make_architecture.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT = Path(__file__).resolve().parent.parent
FIGURES = PROJECT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

BLUE_E, BLUE_F = "#3B6EA5", "#DCE6F2"
RUST_E, RUST_F = "#B3512F", "#F6E4DB"
GREEN_E, GREEN_F = "#4F7A46", "#E2EDDF"
GREY_E, GREY_F = "#8C8C8C", "#EDEDED"
ARROW = "#4A4A4A"
INK = "#1A1A1A"

plt.rcParams.update({"font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, text, edge, fill, *, bold=False, size=10.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.9",
        linewidth=1.4, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, color=INK,
            fontweight="bold" if bold else "normal", zorder=3, linespacing=1.45)


def arrow(ax, x0, y0, x1, y1, *, lw=1.6, color=ARROW):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=15,
        linewidth=lw, color=color, zorder=4,
        shrinkA=0, shrinkB=0))


# ----------------------------------------------------------------- high level
fig, ax = plt.subplots(figsize=(12.6, 7.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

LABEL_X, LABEL_W = 1.5, 15.0
CONTENT_X, CONTENT_W = 20.0, 78.0
CHANNEL = CONTENT_X + CONTENT_W / 2          # the empty vertical channel

BANDS = [
    ("DATA\nLAYER", 79.0, 15.0),
    ("PROCESSING\nLAYER", 55.5, 15.0),
    ("ARTEFACT\nLAYER", 32.0, 15.0),
    ("PRESENTATION\nLAYER", 8.5, 13.0),
]

for name, y, h in BANDS:
    box(ax, LABEL_X, y, LABEL_W, h, name, GREY_E, GREY_F, bold=True, size=10)

# Data layer: three boxes, left to right.
data_y, data_h = 79.0, 15.0
dw = 22.5
gap = (CONTENT_W - 3 * dw) / 2
for i, label in enumerate([
        "SemEval-2018 Task 1\nE-c English corpus\n6,838 / 886 / 3,259",
        "data/interim/\nstandardised CSVs\nnotebook 01",
        "data/processed/\nboth emoji tracks\nnotebook 03"]):
    x = CONTENT_X + i * (dw + gap)
    box(ax, x, data_y, dw, data_h, label, RUST_E, RUST_F)
    if i < 2:
        arrow(ax, x + dw, data_y + data_h / 2, x + dw + gap, data_y + data_h / 2)

# Data -> processing.
arrow(ax, CHANNEL, data_y, CHANNEL, 55.5 + 15.0, lw=1.9)

# Processing layer: the four notebooks.
proc_y, proc_h = 55.5, 15.0
nw = 17.0
ngap = (CONTENT_W - 4 * nw) / 3
for i, label in enumerate([
        "01\ndata load", "02\nEDA", "03\ntrain", "04\nevaluate"]):
    x = CONTENT_X + i * (nw + ngap)
    box(ax, x, proc_y, nw, proc_h, label, BLUE_E, BLUE_F)
    if i < 3:
        arrow(ax, x + nw, proc_y + proc_h / 2, x + nw + ngap, proc_y + proc_h / 2)

# Processing -> artefacts.
arrow(ax, CHANNEL, proc_y, CHANNEL, 32.0 + 15.0, lw=1.9)

# Artefact layer.
art_y, art_h = 32.0, 15.0
aw = 22.5
agap = (CONTENT_W - 3 * aw) / 2
for i, label in enumerate([
        "models/\n4 x .keras\nwritten by 03",
        "results/\nthresholds + 13 tables\nwritten by 03 and 04",
        "figures/\n17 plots\nwritten by 02 and 04"]):
    box(ax, CONTENT_X + i * (aw + agap), art_y, aw, art_h,
        label, GREEN_E, GREEN_F)

# Artefacts -> presentation.
arrow(ax, CHANNEL, art_y, CHANNEL, 8.5 + 13.0, lw=1.9)

pw = 46.0
box(ax, CHANNEL - pw / 2, 8.5, pw, 13.0,
    "Streamlit application\napp/streamlit_app.py", RUST_E, RUST_F, bold=True,
    size=11.5)
ax.text(CHANNEL + pw / 2 + 2.0, 8.5 + 13.0 / 2,
        "reads models/ and results/\ndirectly, with no copying step",
        ha="left", va="center", fontsize=9.5, color="#5A6472", linespacing=1.4)

fig.tight_layout()
out1 = FIGURES / "arch_highlevel.png"
fig.savefig(out1, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved", out1)


# ------------------------------------------------------------ experiment plan
fig, ax = plt.subplots(figsize=(12.6, 7.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

box(ax, 30, 90, 40, 9,
    "SemEval-2018 E-c\n6,838 / 886 / 3,259 tweets", RUST_E, RUST_F, bold=True)

TRACKS = [
    (2.5, "TRACK A: text_no_emoji", "every emoji deleted", BLUE_E, BLUE_F),
    (52.5, "TRACK B: text_with_emoji", "every emoji verbalised", GREEN_E, GREEN_F),
]
TW = 45.0

for x0, title, sub, edge, fill in TRACKS:
    cx = x0 + TW / 2
    arrow(ax, 50, 90, cx, 76 + 9)

    box(ax, x0, 76, TW, 9, f"{title}\n{sub}", edge, fill, bold=True)

    half = (TW - 3.0) / 2
    for j, arch in enumerate(["BiLSTM\n3 configurations",
                              "Transformer\n3 configurations"]):
        bx = x0 + j * (half + 3.0)
        arrow(ax, cx, 76, bx + half / 2, 62 + 10)
        box(ax, bx, 62, half, 10, arch, edge, fill, size=10)
        arrow(ax, bx + half / 2, 62, bx + half / 2, 49 + 9)

    box(ax, x0, 49, TW, 9,
        "Phase 1: grid search, 5 epochs\nselect best on development Micro-F1",
        edge, fill, size=10)
    arrow(ax, cx, 49, cx, 35 + 9)
    box(ax, x0, 35, TW, 9,
        "Phase 2: final training, 10 epochs\nplus decision threshold sweep",
        edge, fill, size=10)
    arrow(ax, cx, 35, cx, 20 + 10)
    box(ax, x0, 20, TW, 10,
        "Micro-F1, Macro-F1, per-class F1\nconfusion matrices, ROC-AUC",
        GREY_E, GREY_F, size=10)

# The comparison, with the label clear of both result boxes.
ax.text(50, 13.5, "delta F1  =  F1(Track B)  -  F1(Track A)",
        ha="center", va="center", fontsize=12, fontweight="bold", color=RUST_E)
ax.add_patch(FancyArrowPatch(
    (47.5, 25.0), (52.5, 25.0), arrowstyle="<|-|>", mutation_scale=15,
    linewidth=2.0, color=RUST_E, zorder=4, shrinkA=0, shrinkB=0))

fig.tight_layout()
out2 = FIGURES / "arch_design.png"
fig.savefig(out2, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved", out2)
