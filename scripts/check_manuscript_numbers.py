#!/usr/bin/env python3
"""Check headline numbers reported in the manuscript against the analysis tables in this repository.

Reads only analysis/tables and analysis/results; it imports no model and recomputes nothing. Each value is
compared after rounding to the precision printed in the manuscript (Table 1, Table 3 and the Results text).
"""
import csv
import json
import statistics
from pathlib import Path

A = Path(__file__).resolve().parents[1] / "analysis"
checks = 0


def rows(name):
    with (A / "tables" / name).open() as f:
        return list(csv.DictReader(f))


def check(label, value, expected, nd):
    global checks
    assert round(float(value), nd) == expected, f"{label}: {value} does not round to {expected}"
    checks += 1


# Table 1: per-fold, two-fold mean +/- sample s.d., and pooled metrics
base = {r["scope"]: r for r in rows("baseline_metrics.csv")}
assert int(base["fold1"]["n"]) == 52792 and int(base["fold2"]["n"]) == 53010 and int(base["pooled"]["n"]) == 105802
checks += 1
table1 = {
    "accuracy": ([0.6361, 0.6099], 0.6230, 0.0185, 0.6230),
    "macro_recall": ([0.4199, 0.4110], 0.4155, 0.0063, 0.4041),
    "macro_precision": ([0.5199, 0.5187], 0.5193, 0.0009, 0.5114),
    "macro_f1": ([0.4339, 0.4305], 0.4322, 0.0024, 0.4286),
}
for metric, (folds, mean, sd, pooled) in table1.items():
    vals = [float(base[s][metric]) for s in ("fold1", "fold2")]
    for v, e in zip(vals, folds):
        check(metric, v, e, 4)
    check(metric + " mean", statistics.mean(vals), mean, 4)
    check(metric + " sd", statistics.stdev(vals), sd, 4)
    check(metric + " pooled", base["pooled"][metric], pooled, 4)

# Whole-input replacement (Results, "Sensitivity to PPL and XPL replacement")
mod = json.loads((A / "results" / "modality_summary.json").read_text())
assert mod["n"] == 706 and mod["classes"] == 25 and mod["larger_XPL_mean_drop_classes"] == 20
checks += 1
check("n-weighted PPL", mod["n_weighted_mean_PPL_drop"], 0.281, 3)
check("n-weighted XPL", mod["n_weighted_mean_XPL_drop"], 0.419, 3)
check("equal-weight PPL", mod["equal_class_mean_PPL_drop"], 0.281, 3)
check("equal-weight XPL", mod["equal_class_mean_XPL_drop"], 0.416, 3)
check("contrast min", mod["class_difference_range"][0], -0.078, 3)
check("contrast max", mod["class_difference_range"][1], 0.535, 3)

# Table 3: method-minus-random contrasts (mean, pointwise and simultaneous 95% CIs, grains above random)
table3 = {
    ("Grad-CAM", "PPL"): (0.005, -0.024, 0.040, -0.047, 0.057, 13),
    ("Grad-CAM", "XPL"): (0.048, 0.006, 0.092, -0.021, 0.117, 20),
    ("Chefer branch", "PPL"): (0.031, -0.006, 0.076, -0.035, 0.098, 16),
    ("Chefer branch", "XPL"): (0.057, 0.002, 0.119, -0.035, 0.150, 19),
    ("Chefer cross", "PPL"): (-0.043, -0.078, -0.011, -0.097, 0.011, 9),
    ("Chefer cross", "XPL"): (0.027, -0.034, 0.090, -0.072, 0.126, 15),
    ("cross-attn", "PPL"): (-0.024, -0.058, 0.013, -0.080, 0.033, 11),
    ("cross-attn", "XPL"): (-0.019, -0.071, 0.028, -0.098, 0.061, 13),
    ("rollout", "PPL"): (-0.031, -0.061, -0.003, -0.078, 0.015, 10),
    ("rollout", "XPL"): (0.015, -0.033, 0.068, -0.065, 0.096, 16),
}
mmr = {(r["method"], r["branch"]): r for r in rows("irof_method_minus_random.csv")}
assert len(mmr) == 10
for key, (m, pl, ph, sl, sh, above) in table3.items():
    r = mmr[key]
    assert int(r["n_grains"]) == 30 and int(r["n_section_keys"]) == 29
    for col, e in [("section_equal_mean", m), ("pointwise95_low", pl), ("pointwise95_high", ph),
                   ("simultaneous95_all41_low", sl), ("simultaneous95_all41_high", sh)]:
        check(f"{key} {col}", r[col], e, 3)
    assert int(r["positive_grains"]) == above
    checks += 1

# Family of 41 contrasts: every simultaneous interval includes zero
fam = rows("irof_all_paired_contrasts.csv")
assert len(fam) == 41
assert all(float(r["simultaneous95_all41_low"]) <= 0 <= float(r["simultaneous95_all41_high"]) for r in fam)
checks += 1

print(f"{checks} checks passed")
