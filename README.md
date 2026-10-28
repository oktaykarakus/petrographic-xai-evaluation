# Evaluating classification and explainability in multimodal petrographic image analysis

Reproducibility materials for the manuscript of the same title by Thomas James Wood, Oktay Karakuş and Can Eyüpoğlu, prepared for *Scientific Reports*.

> **Status: private.** This repository accompanies a manuscript under preparation for submission. Do not make it public, create releases or archive it until the paper has been accepted (see [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md)). No licence has been chosen yet, so no reuse rights are granted at present.

## Purpose

The study re-analyses saved outputs from an earlier evaluation of the released LITHOS dual-stream vision transformer (Chacón *et al.*, NeurIPS 2025). It examines four aspects of the model:

- how its classification errors relate to the petrographic label taxonomy;
- how sensitive its predictions are to replacing the plane-polarised (PPL) or cross-polarised (XPL) input;
- how five attribution procedures differ on identical inputs;
- how those procedures perform under superpixel deletion, with uncertainty estimated by resampling thin sections and simultaneous inference.

This repository contains the saved numerical records, the analysis code that turns them into every statistic in the manuscript, and the scripts that draw the manuscript's analytical figures and supplementary tables. It does **not** contain the LITHOS images, the LITHOS model weights or any newly trained model. The study did not train, fine-tune or re-run the model.

## Relationship to the manuscript

| Manuscript item | Produced by | From |
| --- | --- | --- |
| Table 1, Table 2, Fig. 1 | `analysis/scripts/analyze.py` → `scripts/render_figures.py` | Saved fold-1 and fold-2 predictions |
| Fig. 2, Supplementary Table S1 | `analyze.py` → `render_figures.py`, `make_supplementary_tables.py` | Saved category means of the whole-input replacement |
| Table 3, Figs 4–5, Supplementary Table S2 | `analyze.py` (paired contrasts, section bootstrap, simultaneous intervals) → `render_figures.py`, `make_supplementary_tables.py` | 360 saved per-grain deletion (AOC) scores |
| Fig. 3 | `scripts/compose_figure3.py` (provenance only; see below) | Saved attribution figures containing LITHOS imagery (not distributed) |

## Repository layout

```text
analysis/                 Frozen saved-output reanalysis (copy of the Phase 3B package)
  config/                 Analysis contract: seeds, contrast family, input SHA-256 hashes
  scripts/analyze.py      All statistics, tables and analysis figures; no model is imported
  scripts/verify.py       Independent check with scikit-learn and the standard library
  tables/ results/ figure_source_data/ figures/   Outputs of analyze.py
data/saved_records/       The eight saved records analysed (predictions, category means, AOC scores, run logs)
scripts/                  Manuscript figures, supplementary tables and the headline-number check
figures/                  Manuscript Figures 1, 2, 4 and 5 (PNG and PDF); Figure 3 provenance record
supplementary_tables/     Supplementary Tables S1 and S2
original_experiments/     Code-only copy of the notebook that ran the original experiments
docs/                     Checkpoint and data provenance, and the reproducibility guide
```

## Requirements

The analysis was run with Python 3.12.3, NumPy 1.26.4, Matplotlib 3.9.1, scikit-learn 1.5.1 (used only by `verify.py`) and Pillow 10.4.0 (used only by `compose_figure3.py`). These versions are pinned in [`requirements.txt`](requirements.txt).

```bash
python -m pip install -r requirements.txt
```

`scripts/render_figures.py` sets figures in Arial. If Arial is not installed, Matplotlib substitutes a font and the figures are not byte-identical.

## Reproducing the manuscript results

Run all commands from the repository root. Each of these was run on the staged repository; the outcomes are recorded in [`docs/reproducibility.md`](docs/reproducibility.md).

```bash
python analysis/scripts/analyze.py              # statistics, tables, bootstrap intervals (~20 s)
python analysis/scripts/verify.py               # independent verification of the baseline metrics and all 41 contrasts
python scripts/render_figures.py                # manuscript Figures 1, 2, 4 and 5
python scripts/make_supplementary_tables.py     # Supplementary Tables S1 and S2
python scripts/check_manuscript_numbers.py      # 89 checks of headline numbers against the tables
```

`analyze.py` checks the SHA-256 hash of every input before it computes anything, and stops if an input has changed. Running it rewrites the files in `analysis/`; the bootstrap is seeded (`20261024`), so a rerun reproduces them exactly.

## What cannot be rerun from this repository alone

These steps need the LITHOS dataset, the LITHOS checkpoints and a GPU, and **have not been re-run** for this repository:

- the baseline inference that produced the saved predictions;
- the whole-input replacement experiment;
- the attribution maps and superpixel deletion curves.

The original notebook is provided as code only. Some quantities were never saved: grain-level replacement outcomes, individual random deletion orders, segmentation masks and per-grain probability curves. They cannot be recovered from any distributed material. See [`docs/reproducibility.md`](docs/reproducibility.md) and [`docs/checkpoint_provenance.md`](docs/checkpoint_provenance.md).

## External resources

- **LITHOS dataset:** Kaggle *LITHOS-DATASET*, version 1, CC BY-NC-SA 4.0 — <https://www.kaggle.com/datasets/f1caa92f0ca6bace86fc8fa6d557d3cac4d2cfe545f5aeb7ea2f141ab68d3720>
- **LITHOS code:** <https://github.com/BCV-Uniandes/LITHOS>, commit `ac3391035ea86b9e5d744d0974a061052d033602`, MIT licence
- **LITHOS models:** distributed through the link in the LITHOS publication and repository README, under the licence stated by the LITHOS authors. The specific checkpoints used here are described in [`docs/checkpoint_provenance.md`](docs/checkpoint_provenance.md).

Obtain these resources from their official sources and follow their licences. They are not redistributed here.

## Citation

Please cite the article once it is published. A [`CITATION.cff`](CITATION.cff) file is provided; journal, volume and DOI will be added on publication. Please also cite the LITHOS publication:

> Chacón, I. D., Ruiz Puentes, P., Pearse, J. & Arbeláez, P. Towards automated petrography. In *Advances in Neural Information Processing Systems* **38** (2025). https://doi.org/10.52202/085713-3740

## Contact

Corresponding author: Oktay Karakuş, Department of Computer Science and Informatics, Cardiff University. *[Contact e-mail to be added.]*
