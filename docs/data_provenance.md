# Data provenance

## External, not redistributed

| Resource | Source | Licence | Used for |
| --- | --- | --- | --- |
| LITHOS images, annotations and fold assignments | Kaggle *LITHOS-DATASET*, version 1 (owner: Paola Ruiz Puentes). Fold files `Fold1_complete_info_allclasses.csv`, `Fold2_complete_info_allclasses.csv` | CC BY-NC-SA 4.0 | All original experiments |
| LITHOS code | github.com/BCV-Uniandes/LITHOS, commit `ac3391035ea86b9e5d744d0974a061052d033602` | MIT | Model architecture, data loader, label mapping |
| LITHOS trained models | Link in the LITHOS publication and README | CC BY-NC-SA 4.0, as stated in the LITHOS publication | Inference in the original experiments |

## Saved records included (`data/saved_records/`)

These are outputs of the original experiments, copied byte-for-byte. The directory layout follows the original project (`code/pipeline/...`) so the hash checks in `analysis/config/analysis_config.json` apply unchanged.

| File | SHA-256 | Content |
| --- | --- | --- |
| `code/pipeline/cache/preds_mc25_b_fold1.npz` | `204312ad…8a3d` | `y_true` and `y_pred` for 52,792 fold-1 grains (checkpoint B) |
| `code/pipeline/cache/preds_mc25_a_fold2.npz` | `7e1132bf…025a` | `y_true` and `y_pred` for 53,010 fold-2 grains (checkpoint A) |
| `code/pipeline/cache/modality_reliance_25class.csv` | `deb33036…1a25b` | Category means of the probability reduction after PPL or XPL replacement (706 grains, 25 categories) |
| `code/pipeline/cache/irof_aoc_per_grain.csv` | `2be4c782…dbc41` | 360 AOC scores: 30 grains × 2 branches × (5 procedures + random reference) |
| `code/pipeline/runs/2026-08-20_baseline-ranking-metrics.json` | `d80a1a25…9163` | Historical probability-based metrics (recorded only; no aligned probabilities exist) |
| `code/pipeline/runs/2026-07-{14,20,29}_irof-fidelity.json` | see `analysis/config/analysis_config.json` | Historical rounded aggregate deletion curves (used only as consistency records) |

Full hashes are in `analysis/config/analysis_config.json` and `analysis/results/input_manifest.json`.

**Content derived from LITHOS.**
- `y_true` holds the LITHOS expert annotations, mapped to 25 categories, in fold order without grain identifiers.
- `irof_aoc_per_grain.csv` identifies grains by their LITHOS image path (for example `GMAS_LOTE/333/154/ppl-0.png`). The thin-section key is taken from this path.
- No image pixels are included.

These records are adapted from a CC BY-NC-SA 4.0 dataset; see the licence note in the README.

## Provenance-only inputs (not included)

| File | SHA-256 | Reason |
| --- | --- | --- |
| `code/pipeline/pipelineLithosXAI.ipynb` | `e4fe8b99…7b3e` | Its saved outputs embed LITHOS imagery. A code-only copy is in `original_experiments/`. |
| `code/external/lithos/polar_vit_arch.py` | `1c9900de…e55` | Third-party code. Use the pinned upstream commit. |

`analyze.py` hash-checks these files only if they are present.

## Not retained by the original experiments

These were never saved, so they cannot be provided:
- grain identifiers and class probabilities aligned with the baseline predictions;
- grain-level outcomes and thin-section membership for the whole-input replacement;
- individual random deletion orders, per-grain probability curves, segment areas and segmentation masks;
- the identities of the 706 replacement grains, and their overlap with the deletion sample.

## Figure 3

Figure 3 shows input patches from the LITHOS dataset with attribution overlays. It was assembled by `scripts/compose_figure3.py`, which crops panels 1:1, without resampling, from five figures saved by the original notebook. The source hashes, grain identifiers and output hash are in `figures/Figure3_Qualitative_Attribution.provenance.json`. The image itself is not included in this repository.
