"""Write presentation copies of Supplementary Tables S1 and S2 from the frozen Phase 3B tables.

Columns are renamed and values rounded to four decimals for readers; internal identifiers and
workflow columns are dropped. The full-precision frozen tables remain the reference records.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repository root (paths adapted from Phase3C/Revision3/scripts)
T = ROOT / "analysis" / "tables"
OUT = ROOT / "supplementary_tables"
OUT.mkdir(exist_ok=True)
NAME = {"Grad-CAM": "Grad-CAM", "Chefer branch": "Chefer branch adaptation", "Chefer cross": "Chefer cross adaptation",
        "cross-attn": "Cross-attention", "rollout": "Rollout", "random": "Random order"}
KIND = {"method_minus_random": "Procedure minus random order", "random_adjusted_branch_interaction": "Branch contrast (XPL minus PPL, random-adjusted)",
        "paired_method_comparison": "Pairwise procedure comparison", "raw_branch_difference": "Raw XPL minus PPL AOC"}
r4 = lambda v: f"{float(v):.4f}"

rows = list(csv.DictReader((T / "irof_all_paired_contrasts.csv").open()))
assert len(rows) == 41
with (OUT / "Supplementary_Table_S2_Deletion_Contrasts.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Contrast type", "Procedure", "Comparator", "Branch", "Grains", "Thin sections", "Section-equal mean", "Grain-equal mean",
                "Section SD", "Standardised effect", "Pointwise 95% CI lower", "Pointwise 95% CI upper", "Simultaneous 95% CI lower",
                "Simultaneous 95% CI upper", "Grains positive", "Grains zero", "Grains negative", "Grain median", "Grain minimum", "Grain maximum"])
    for r in rows:
        comp = {"recorded random mean": "Random order", "XPL delta minus PPL delta": "—", "raw XPL AOC minus raw PPL AOC": "—"}.get(r["comparator"], NAME.get(r["comparator"], r["comparator"]))
        br = "XPL minus PPL" if r["branch"] == "XPL-PPL" else r["branch"]
        w.writerow([KIND[r["kind"]], NAME[r["method"]], comp, br, r["n_grains"], r["n_section_keys"], r4(r["section_equal_mean"]), r4(r["grain_equal_mean"]),
                    r4(r["section_sd"]), r4(r["paired_standardized_section_effect_dz"]), r4(r["pointwise95_low"]), r4(r["pointwise95_high"]),
                    r4(r["simultaneous95_all41_low"]), r4(r["simultaneous95_all41_high"]), r["positive_grains"], r["zero_grains"], r["negative_grains"],
                    r4(r["grain_median"]), r4(r["grain_min"]), r4(r["grain_max"])])

rows = list(csv.DictReader((T / "modality_class_summary.csv").open()))
assert len(rows) == 25
with (OUT / "Supplementary_Table_S1_Modality_Sensitivity.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Category", "Grains", "Mean reduction, PPL replaced", "Mean reduction, XPL replaced", "XPL minus PPL"])
    for r in rows:
        w.writerow([r["class_name"], r["n_selected_correct_grains"], r4(r["mean_drop_PPL_filled"]), r4(r["mean_drop_XPL_filled"]),
                    r4(r["paired_difference_of_class_means_XPL_minus_PPL"])])
print("written", sorted(p.name for p in OUT.iterdir()))
