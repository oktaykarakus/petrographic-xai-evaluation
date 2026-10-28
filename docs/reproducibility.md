# Reproducibility guide

## Verified from this repository

These commands were run from the repository root on 28 October 2026 with Python 3.12.3, NumPy 1.26.4, Matplotlib 3.9.1, scikit-learn 1.5.1 and Pillow 10.4.0.

| Step | Command | Verified outcome |
| --- | --- | --- |
| 1 | `python analysis/scripts/analyze.py` | All distributed inputs pass their SHA-256 checks. Compared with the frozen analysis package of 24 October 2026, 49 of 51 outputs are identical (tables, figure source data, analysis figures and results; the bootstrap NPZ is equal array by array). In the other two, `results/input_manifest.json` and `results/historical_probability_metrics.json`, only path fields differ: they now hold repository-relative paths and mark the two provenance-only inputs as absent. |
| 2 | `python analysis/scripts/verify.py` | scikit-learn reproduces all baseline metrics and confusion matrices to within 1e-12. The standard library reproduces all 41 contrast means (maximum error 2e-17). None of the 41 simultaneous intervals excludes zero. |
| 3 | `python scripts/render_figures.py` | Figures 1, 2, 4 and 5 (PNG and PDF) are byte-identical to the manuscript figures. The script asserts that every plotted mean matches the analysis tables. |
| 4 | `python scripts/make_supplementary_tables.py` | Supplementary Tables S1 and S2 are byte-identical to the manuscript versions. |
| 5 | `python scripts/check_manuscript_numbers.py` | 89 checks pass, covering Table 1, the replacement summaries (706 grains; XPL larger in 20 of 25 categories), the Table 3 contrasts and intervals for 30 grains in 29 thin sections, and all 41 simultaneous intervals including zero. |

Table 2 and the confusion percentages in the Results text are read directly from `analysis/tables/baseline_confusion_pooled.csv` and `baseline_per_class_metrics.csv`; no separate script produces them. The absolute AOC means in Table 3 come from `analysis/tables/irof_recorded_and_derived_grain_scores.csv`.

## Coverage by analysis

| Analysis | Reproducible here | Needs external LITHOS inputs | Saved outputs only |
| --- | --- | --- | --- |
| Classification metrics, confusion structure (Tables 1–2, Fig. 1) | Yes, from the saved predictions | Regenerating the predictions needs the dataset, checkpoints A and B, and inference code (not included) | Probability-based metrics (no aligned probabilities) |
| Whole-input replacement (Fig. 2, Table S1) | Summaries and figure, from the category means | Re-running needs the dataset, checkpoint A, a GPU and the notebook | Grain-level outcomes were not saved, so no uncertainty can be estimated |
| Superpixel deletion statistics (Table 3, Figs 4–5, Table S2) | Yes: contrasts, section bootstrap, simultaneous intervals and sensitivity analyses, from the 360 saved scores | Recomputing the scores needs the dataset, checkpoint A, a GPU and the notebook | Random reference saved as a mean over three orders only |
| Attribution maps (Fig. 3) | No | Regeneration would require new inference, which is outside the scope of the study | Fig. 3 panels come from saved figures; provenance in `figures/` |

## Re-running the original experiments

`original_experiments/pipelineLithosXAI_code_only.ipynb` contains the code that produced the whole-input replacement, attribution and deletion records. Its outputs have been removed. It expects:

- the LITHOS dataset, set by the `LITHOS_DATA` environment variable;
- the LITHOS checkpoints, set by `LITHOS_WEIGHTS`;
- the LITHOS code at commit `ac33910`;
- a CUDA GPU.

The notebook also reads `preds_mc25_a_fold2.npz` to select correctly classified grains. **None of this has been re-run for this repository.** Even with the same inputs, exact agreement is not guaranteed: the scikit-image and pytorch-grad-cam versions were not recorded, and GPU arithmetic is non-deterministic. The checkpoint caveats in [`checkpoint_provenance.md`](checkpoint_provenance.md) also apply.

The baseline predictions were produced by the baseline-inference stage of a separate dissertation notebook, which is not included. The predictions reproduce the published LITHOS metrics. The official LITHOS evaluation code (`main.py` at the pinned commit) evaluates each fold with the model trained on the other fold.
