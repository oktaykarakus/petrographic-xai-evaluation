"""Compose main-text Figure 3 (qualitative attribution comparison) from saved notebook figures.

Pixels are copied 1:1 from the five per-method figures saved by the original analysis
(dissertation/code/figures/xai/stage*.png, read-only); there is no resampling, interpolation,
sharpening or colour change, and no attribution value is regenerated. These figures hold the same two
grains as stage4_fidelity_ladder_grid.png at a higher saved resolution (267 x 266 px per panel against
about 203 px); the script verifies panel-by-panel agreement with that grid (Pearson r >= 0.99 after
downsampling) before composing. The output carries no rights banner. Its rights status (NOT CLEARED until written permission from the
LITHOS rights holders is obtained) is recorded in the provenance file and figures/Figure_Manifest.md.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Paths adapted from Phase3C/Revision3/scripts. The source figures contain LITHOS imagery and are not distributed;
# set THOM_XAI_FIGURES to a permitted local copy of dissertation/code/figures/xai.
XAI = Path(os.environ.get("THOM_XAI_FIGURES", "xai_figures_not_distributed"))
OUT = Path(__file__).resolve().parents[1] / "figures"

SOURCES = {  # method label -> saved per-method figure (same six exemplar rows in each)
    "Grad-CAM": "stage2_gradcam.png",
    "Chefer branch": "stage4b_chefer_branch.png",
    "Chefer cross": "stage4_chefer_cross.png",
    "Cross-attention": "stage3a_fusion_crossattn.png",
    "Rollout": "stage3b_rollout.png",
}
EXPECTED_SHA = {
    "stage4_fidelity_ladder_grid.png": "152d2df01aace45e9c77e54f0a5b5ab94187509114ce0eecfbab47d6831760b1",
}
# Panel geometry of the per-method figures (detected from the saved PNGs; 4 columns x 6 rows)
PCOL = [(10, 277), (305, 572), (600, 867), (896, 1163)]  # raw PPL, overlay PPL, raw XPL, overlay XPL
PROW = [(26, 292), (324, 590), (621, 887), (919, 1185), (1216, 1483), (1514, 1780)]
GRAINS = [("a", "Monocrystalline", 0), ("b", "Polycrystalline", 2)]  # exemplar rows 1 and 3
# Grid geometry, used only for the agreement check
GCOL = [(28, 233), (249, 455), (470, 676), (691, 897)]
GROW = {"raw": (75, 281), "Grad-CAM": (293, 499), "Cross-attention": (511, 716), "Rollout": (728, 934),
        "Chefer cross": (946, 1152), "Chefer branch": (1164, 1370)}

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before = {f: sha(XAI / f) for f in list(SOURCES.values()) + ["stage4_fidelity_ladder_grid.png"]}
assert before["stage4_fidelity_ladder_grid.png"] == EXPECTED_SHA["stage4_fidelity_ladder_grid.png"]
src = {m: Image.open(XAI / f).convert("RGB") for m, f in SOURCES.items()}
grid = Image.open(XAI / "stage4_fidelity_ladder_grid.png").convert("RGB")


def panel(img, row, col):
    (x0, x1), (y0, y1) = PCOL[col], PROW[row]
    return img.crop((x0, y0, x1, y1))


# Agreement with the grid figure (grain a -> grid cols 0,1; grain b -> grid cols 2,3)
check = []
for gi, (_, _, row) in enumerate(GRAINS):
    for bi, (raw_col, ov_col) in enumerate([(0, 1), (2, 3)]):
        gcol = GCOL[2 * gi + bi]
        for name in ["raw"] + list(SOURCES):
            p = panel(src["Grad-CAM" if name == "raw" else name], row, raw_col if name == "raw" else ov_col)
            (y0, y1) = GROW[name]
            g = grid.crop((gcol[0] + 2, y0 + 2, gcol[1] - 2, y1 - 2))
            r = np.corrcoef(np.asarray(p.resize(g.size, Image.LANCZOS), float).ravel(), np.asarray(g, float).ravel())[0, 1]
            check.append(r)
            assert r >= 0.99, (name, gi, bi, r)
# Input panels must be identical in every per-method figure
for _, _, row in GRAINS:
    for c in (0, 2):
        ref = np.asarray(panel(src["Grad-CAM"], row, c), int)
        for m in SOURCES:
            assert np.abs(np.asarray(panel(src[m], row, c), int) - ref).max() <= 2, (m, row, c)

# Layout: rows = grain x branch, columns = input + five procedures (Table 3 order), with a two-level
# header separating the class-targeted procedures from the class-agnostic baselines
P_W, P_H = 267, 266
COLS = ["Input"] + list(SOURCES)
HEAD = {"Input": "Model input", "Grad-CAM": "Grad-CAM", "Chefer branch": "Chefer branch\nadaptation",
        "Chefer cross": "Chefer cross\nadaptation", "Cross-attention": "Cross-attention", "Rollout": "Rollout"}
GROUPS = [("Class-targeted", 1, 3), ("Class-agnostic", 4, 5)]
F = "/System/Library/Fonts/Supplemental/Arial.ttf"
FB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
# 30 px is about 8.4 pt when the figure is printed 180 mm wide (about 255 dpi)
f_head, f_group, f_row, f_grain, f_let = (ImageFont.truetype(FB, 30), ImageFont.truetype(F, 30),
                                          ImageFont.truetype(FB, 30), ImageFont.truetype(F, 30),
                                          ImageFont.truetype(FB, 40))
LEFT, TOP, GAP, ROW_GAP, INPUT_GAP, GRAIN_GAP, BOTTOM, TICK = 150, 150, 6, 10, 24, 40, 14, 7
W = LEFT + 6 * P_W + 4 * GAP + INPUT_GAP + 8
H = TOP + 4 * P_H + 2 * ROW_GAP + GRAIN_GAP + BOTTOM
canvas = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(canvas)


def cx(k):
    return LEFT + k * P_W + max(0, k - 1) * GAP + (INPUT_GAP if k >= 1 else 0)


def ry(i):
    return TOP + i * (P_H + ROW_GAP) + (GRAIN_GAP - ROW_GAP if i >= 2 else 0)


for k, c in enumerate(COLS):
    draw.multiline_text((cx(k) + P_W // 2, TOP - 10), HEAD[c], font=f_head, fill="black", anchor="md",
                        align="center", spacing=3)
for label, k0, k1 in GROUPS:
    x0, x1, y = cx(k0) + 4, cx(k1) + P_W - 4, TOP - 86
    draw.line((x0, y, x1, y), fill="black", width=2)
    draw.text(((x0 + x1) // 2, y - 8), label, font=f_group, fill="black", anchor="md")

i = 0
for letter, cat, row in GRAINS:
    top = ry(i)
    for bi, (branch, raw_col, ov_col) in enumerate([("PPL", 0, 1), ("XPL", 2, 3)]):
        y = ry(i)
        for k, c in enumerate(COLS):
            p = panel(src["Grad-CAM"], row, raw_col) if c == "Input" else panel(src[c], row, ov_col)
            canvas.paste(p, (cx(k), y))  # 1:1 copy
        draw.text((LEFT - 14, y + P_H // 2), branch, font=f_row, fill="black", anchor="rm")
        # patch-centre ticks outside the input panel: right edge (vertical centre) and the outer edge
        # of the grain block (horizontal centre), so ticks of adjacent rows never meet
        x0, xm, ym = cx(0), cx(0) + P_W // 2, y + P_H // 2
        draw.line((x0 + P_W + 1, ym, x0 + P_W + TICK, ym), fill="black", width=3)
        if bi == 0:
            draw.line((xm, y - TICK, xm, y - 2), fill="black", width=3)
        else:
            draw.line((xm, y + P_H + 1, xm, y + P_H + TICK), fill="black", width=3)
        i += 1
    mid = (top + ry(i - 1) + P_H) // 2
    draw.text((12, top - 6), letter, font=f_let, fill="black", anchor="lt")
    lab = Image.new("RGBA", (2 * P_H, 44), (255, 255, 255, 0))
    ImageDraw.Draw(lab).text((P_H, 22), cat, font=f_grain, fill="black", anchor="mm")
    lab = lab.rotate(90, expand=True)
    canvas.paste(lab, (34, mid - lab.size[1] // 2), lab)

name = "Figure3_Qualitative_Attribution"
OUT.mkdir(exist_ok=True)
out = OUT / f"{name}.png"
canvas.save(out, dpi=(255, 255))
after = {f: sha(XAI / f) for f in before}
assert after == before, "source figure changed"
meta = {
    "sources": {f: before[f] for f in SOURCES.values()},
    "agreement_reference": {"stage4_fidelity_ladder_grid.png": before["stage4_fidelity_ladder_grid.png"]},
    "agreement_min_pearson_r": round(float(min(check)), 4), "agreement_panels_checked": len(check),
    "grains": {"a": "GMAS_LOTE2/Muestra366/170 (Monocrystalline; exemplar row 1)",
               "b": "GMAS_LOTE/486/181 (Polycrystalline; exemplar row 3)"},
    "panel_px": [P_W, P_H], "resampling": "none (crop and paste at 1:1)",
    "column_order": COLS, "row_order": ["a PPL", "a XPL", "b PPL", "b XPL"],
    "annotations": "column and group headers, row labels, panel letters, centre ticks outside the input panels",
    "rights_status": "NOT CLEARED: derived from LITHOS imagery (CC BY-NC-SA 4.0); written permission required before any release",
    "output": out.name, "output_size_px": canvas.size,
    "width_at_180mm_dpi": round(canvas.size[0] / (180 / 25.4), 1),
    "output_sha256": sha(out),
}
out.with_suffix(".provenance.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
