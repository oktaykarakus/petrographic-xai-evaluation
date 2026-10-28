"""Render presentation copies of Figures 1, 2, 4 and 5 from the frozen figure source data.

Reads Phase3B/figure_source_data/*.csv (read-only) and writes to Revision2/figures/. Every plotted
value is taken from those tables; no statistic is recalculated except the per-grain branch
interaction in Fig. 5b, which is the within-grain difference of two frozen columns and is checked
against the frozen grain-equal means. Jitter on the vertical axis is display-only and uses the frozen
plot seed. Category order follows petrographic grouping (display only); the frozen Phase 3B figure
files are not touched.
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]  # repository root (paths adapted from Phase3C/Revision3/scripts)
SRC = ROOT / "analysis" / "figure_source_data"
CFG = json.loads((ROOT / "analysis" / "config" / "analysis_config.json").read_text())
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

matplotlib.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})
MM = 1 / 25.4
PPL_C, XPL_C, INT_C, INK, GRID = "#0072B2", "#D55E00", "#009E73", "#222222", "#E6E6E6"


def read(name):
    with (SRC / name).open() as f:
        return list(csv.DictReader(f))


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600, facecolor="white",
                    metadata={"CreationDate": None, "ModDate": None} if ext == "pdf" else None)
    plt.close(fig)


def panel_label(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")


# Petrographic display order with the groups referred to in the text (display only)
GROUPS = [
    ("Quartz family", ["Monocrystalline", "Polycrystalline", "Quartz"]),
    (None, ["Rock fragment"]),
    ("Feldspars", ["Feldspar", "Plagioclase", "Microcline", "Sanidine"]),
    ("Micas", ["Mica", "Muscovite"]),
    ("Amphiboles", ["Amphibole", "Hornblende"]),
    (None, ["Pyroxene", "Opaque", "Heavy mineral"]),
    ("Carbonates", ["Calcite", "Dolomite", "Crystalline mosaic"]),
    ("Biogenic", ["Foraminifer", "Fossil fragment", "Calcareous fossil", "Red algae", "Coral", "Echinoderm"]),
    (None, ["Other (merged)"]),
]
ORDER = [c for _, cs in GROUPS for c in cs]
assert len(ORDER) == 25 and len(set(ORDER)) == 25

# ---------------------------------------------------------------- Figure 1
perf = read("F1_class_performance.csv")
conf = [r for r in read("F1_confusion.csv") if r["scope"] == "pooled"]
names = {r["class_name"] for r in perf}
assert names == set(ORDER), names ^ set(ORDER)
idx = {c: i for i, c in enumerate(ORDER)}
M = np.zeros((25, 25))
for r in conf:
    M[idx[r["true_class_name"]], idx[r["predicted_class_name"]]] = float(r["fraction_of_true_class"])
assert np.allclose(M.sum(1), 1.0)
f1 = {(r["scope"], r["class_name"]): float(r["f1"]) for r in perf}
support = {r["class_name"]: int(r["support"]) for r in perf if r["scope"] == "pooled"}
assert sum(support.values()) == 105802

fig = plt.figure(figsize=(180 * MM, 128 * MM))
ax = fig.add_axes([0.145, 0.20, 0.46, 0.74])
cax = fig.add_axes([0.615, 0.20, 0.010, 0.74])
bx = fig.add_axes([0.74, 0.20, 0.17, 0.74])
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
ax.set_xticks(range(25), ORDER, rotation=90)
ax.set_yticks(range(25), ORDER)
ax.set_xlabel("Predicted category")
ax.set_ylabel("True category")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(True)
start = 0
for g, cs in GROUPS:
    if g:
        ax.add_patch(Rectangle((start - .5, start - .5), len(cs), len(cs), fill=False, ec="#C0392B", lw=0.9))
    start += len(cs)
cb = fig.colorbar(im, cax=cax)
cb.set_label("Fraction of true category", labelpad=3)
cb.outline.set_linewidth(0.5)
panel_label(ax, "a", x=-0.25)

y = np.arange(25)
for scope, off, col, mk, lab in [("fold1", -.22, "#56B4E9", "s", "Fold 1"), ("fold2", .22, "#E69F00", "^", "Fold 2"),
                                 ("pooled", 0, INK, "o", "Pooled")]:
    bx.scatter([f1[(scope, c)] for c in ORDER], y + off, s=9 if scope != "pooled" else 12, marker=mk,
               color=col, label=lab, zorder=3, linewidths=0)
bx.set_ylim(24.6, -0.6)
bx.set_yticks(y, [])
bx.set_xlim(0, 1)
bx.set_xticks([0, .25, .5, .75, 1], ["0", "0.25", "0.5", "0.75", "1"])
bx.set_xlabel("Per-category F1")
bx.grid(axis="x", color=GRID, lw=0.5)
bx.set_axisbelow(True)
for i, c in enumerate(ORDER):
    bx.text(1.03, i, f"{support[c]:,}", transform=bx.get_yaxis_transform(), va="center", ha="left", fontsize=6, color="#444444")
bx.text(1.03, -1.2, "n", transform=bx.get_yaxis_transform(), ha="left", va="center", fontsize=6.5, style="italic")
bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False, handletextpad=0.2, columnspacing=0.8)
panel_label(bx, "b", x=-0.06)
save(fig, "Figure1_Classification_Error_Structure")

# ---------------------------------------------------------------- Figure 2
mod = {r["class_name"]: r for r in read("F2_modality.csv")}
assert set(mod) == set(ORDER)
a = np.array([float(mod[c]["mean_drop_PPL_filled"]) for c in ORDER])
b = np.array([float(mod[c]["mean_drop_XPL_filled"]) for c in ORDER])
d = np.array([float(mod[c]["paired_difference_of_class_means_XPL_minus_PPL"]) for c in ORDER])
assert np.allclose(d, b - a) and int((d > 0).sum()) == 20
n = [int(mod[c]["n_selected_correct_grains"]) for c in ORDER]
assert sum(n) == 706

fig = plt.figure(figsize=(180 * MM, 118 * MM))
ax = fig.add_axes([0.20, 0.10, 0.46, 0.82])
bx = fig.add_axes([0.71, 0.10, 0.26, 0.82])
ax.hlines(y, np.minimum(a, b), np.maximum(a, b), color="#B0B0B0", lw=0.8, zorder=1)
ax.scatter(a, y, s=14, color=PPL_C, marker="o", label="PPL replaced", zorder=3, linewidths=0)
ax.scatter(b, y, s=14, color=XPL_C, marker="s", label="XPL replaced", zorder=3, linewidths=0)
ax.set_yticks(y, [f"{c}  ({k})" for c, k in zip(ORDER, n)])
ax.set_ylim(24.6, -0.6)
ax.set_xlim(0, 0.68)
ax.set_xlabel("Mean reduction in true-class probability")
ax.grid(axis="x", color=GRID, lw=0.5)
ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handletextpad=0.2)
panel_label(ax, "a", x=-0.30)
bx.axvline(0, color="#777777", lw=0.6)
bx.hlines(y, 0, d, color=[XPL_C if v > 0 else PPL_C for v in d], lw=1.6)
bx.set_yticks(y, [])
bx.set_ylim(24.6, -0.6)
bx.set_xlim(-0.12, 0.58)
bx.set_xlabel("XPL minus PPL reduction")
bx.grid(axis="x", color=GRID, lw=0.5)
bx.set_axisbelow(True)
start = 0
for g, cs in GROUPS:
    start += len(cs)
    if start < 25:
        for axx in (ax, bx):
            axx.axhline(start - .5, color="#DDDDDD", lw=0.5, zorder=0)
panel_label(bx, "b", x=-0.03)
save(fig, "Figure2_Modality_Sensitivity")

# ---------------------------------------------------------------- Figures 4 and 5 (deletion)
METHODS = ["Grad-CAM", "Chefer branch", "Chefer cross", "cross-attn", "rollout"]
SHOW = {"Grad-CAM": "Grad-CAM", "Chefer branch": "Chefer branch\n(adaptation)", "Chefer cross": "Chefer cross\n(adaptation)",
        "cross-attn": "Cross-attention", "rollout": "Rollout"}
SHORT = {"Grad-CAM": "GC", "Chefer branch": "CB", "Chefer cross": "CC", "cross-attn": "CA", "rollout": "RO"}
grain_rows = read("F3_grain_deltas.csv")
assert grain_rows == read("F4_grain_deltas.csv")
grains = list(dict.fromkeys(r["sec_patch"] for r in grain_rows))
assert len(grains) == 30
cat = {r["sec_patch"]: r["class_name"] for r in grain_rows}
delta = {(r["sec_patch"], r["method"], r["branch"]): float(r["method_minus_recorded_random"]) for r in grain_rows}
assert len(delta) == 300
ivals = {(r["method"], r["branch"]): r for r in read("F3_intervals.csv")}
inter = {r["method"]: r for r in read("F4_branch_interactions.csv")}
DEL_ORDER = ["Monocrystalline", "Polycrystalline", "Quartz", "Rock fragment", "Feldspar", "Foraminifer"]
assert set(cat.values()) == set(DEL_ORDER)

fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 82 * MM), sharex=True, sharey=True)
rng = np.random.default_rng(CFG["plot_seed"])
for ax, br, col, mk in zip(axes, ["PPL", "XPL"], [PPL_C, XPL_C], ["o", "s"]):
    for i, m in enumerate(METHODS):
        v = np.array([delta[(g, m, br)] for g in grains])
        r = ivals[(m, br)]
        assert abs(v.mean() - float(r["grain_equal_mean"])) < 1e-12
        ax.scatter(v, i + rng.uniform(-.16, .16, len(v)), s=7, color=col, alpha=.45, marker=mk, linewidths=0, zorder=2)
        ax.hlines(i, float(r["simultaneous95_all41_low"]), float(r["simultaneous95_all41_high"]), color="#8C8C8C", lw=0.9, zorder=3)
        ax.hlines(i, float(r["pointwise95_low"]), float(r["pointwise95_high"]), color=col, lw=3.2, zorder=4)
        ax.plot(float(r["section_equal_mean"]), i, "o", ms=3.6, color="black", zorder=5)
    ax.axvline(0, color="#555555", ls=(0, (3, 2)), lw=0.7)
    ax.set_yticks(range(5), [SHOW[m] for m in METHODS])
    ax.set_ylim(4.5, -0.5)
    ax.set_title(f"{br} branch", fontweight="bold")
    ax.set_xlabel("Method AOC minus random-order AOC")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
axes[1].tick_params(labelleft=True)
panel_label(axes[0], "a", x=-0.30)
panel_label(axes[1], "b", x=-0.30)
fig.subplots_adjust(left=0.14, right=0.985, bottom=0.17, top=0.90, wspace=0.42)
save(fig, "Figure4_Deletion_Contrasts")

# Figure 5: grain-level heatmap grouped by category, and per-grain branch interactions
order = [g for c in DEL_ORDER for g in grains if cat[g] == c]
cols = [(m, br) for m in METHODS for br in ("PPL", "XPL")]
G = np.array([[delta[(g, m, br)] for (m, br) in cols] for g in order])
lim = float(np.abs(G).max())
fig = plt.figure(figsize=(180 * MM, 112 * MM))
ax = fig.add_axes([0.13, 0.15, 0.38, 0.78])
cax = fig.add_axes([0.525, 0.15, 0.010, 0.78])
bx = fig.add_axes([0.71, 0.15, 0.27, 0.78])
im = ax.imshow(G, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
ax.set_xticks(range(10), [f"{SHORT[m]}\n{br}" for m, br in cols])
ax.set_yticks([2 + 5 * k for k in range(6)], DEL_ORDER)
ax.tick_params(axis="y", length=0)
for k in range(1, 6):
    ax.axhline(5 * k - .5, color="white", lw=1.6)
for k in range(1, 5):
    ax.axvline(2 * k - .5, color="white", lw=0.8)
ax.set_xlabel("Attribution procedure and branch")
ax.set_ylabel("Grains, grouped by category (five per category)")
cb = fig.colorbar(im, cax=cax)
cb.set_label("Method AOC minus random-order AOC", labelpad=3)
cb.outline.set_linewidth(0.5)
panel_label(ax, "a", x=-0.30)
rng2 = np.random.default_rng(CFG["plot_seed"] + 1)
for i, m in enumerate(METHODS):
    v = np.array([delta[(g, m, "XPL")] - delta[(g, m, "PPL")] for g in grains])
    r = inter[m]
    assert abs(v.mean() - float(r["grain_equal_mean"])) < 1e-12
    bx.scatter(v, i + rng2.uniform(-.16, .16, len(v)), s=7, color="#7F7F7F", alpha=.5, linewidths=0, zorder=2)
    bx.hlines(i, float(r["simultaneous95_all41_low"]), float(r["simultaneous95_all41_high"]), color="#8C8C8C", lw=0.9, zorder=3)
    bx.hlines(i, float(r["pointwise95_low"]), float(r["pointwise95_high"]), color=INT_C, lw=3.2, zorder=4)
    bx.plot(float(r["section_equal_mean"]), i, "o", ms=3.6, color="black", zorder=5)
bx.axvline(0, color="#555555", ls=(0, (3, 2)), lw=0.7)
bx.set_yticks(range(5), [SHOW[m] for m in METHODS])
bx.set_ylim(4.5, -0.5)
bx.set_xlabel("XPL contrast minus PPL contrast")
bx.grid(axis="x", color=GRID, lw=0.5)
bx.set_axisbelow(True)
panel_label(bx, "b", x=-0.36)
save(fig, "Figure5_Grain_Variability")
print("figures written to", OUT)
