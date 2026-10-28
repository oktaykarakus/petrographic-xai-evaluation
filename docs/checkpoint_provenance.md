# Checkpoint provenance

All analyses use the LITHOS dual-stream (PPL + XPL) vision-transformer baseline with 25 output categories, developed and trained by Chacón *et al.* The authors of this study did not train, fine-tune or modify any model.

## Checkpoints used

| Label | Original identifier | Local filename in the original experiments | Trained on | Evaluated on |
| --- | --- | --- | --- | --- |
| A | `LITHOS_BASELINE_FOLD1_25_CLASSES` | `polarvit_multiclass25_a.pth` | fold 1 | fold 2 |
| B | `LITHOS_BASELINE_FOLD2_25_CLASSES` | `polarvit_multiclass25_b.pth` | fold 2 | fold 1 |

| Analysis | Checkpoint and fold | Saved record |
| --- | --- | --- |
| Baseline predictions (Tables 1–2, Fig. 1) | A on fold 2; B on fold 1 | `preds_mc25_a_fold2.npz`, `preds_mc25_b_fold1.npz` |
| Whole-input replacement (Fig. 2, Table S1) | A on fold 2 | `modality_reliance_25class.csv` |
| Attribution and superpixel deletion (Figs 3–5, Table 3, Table S2) | A on fold 2 | `irof_aoc_per_grain.csv`; saved attribution figures |

The original experiments also ran a latent-representation probe with checkpoint A on fold 2. It is not reported in the manuscript, and its outputs are not included here.

The files `best_epoch_f1_fold1.pth` and `best_epoch_f1_fold2.pth` were **not** used.

## Evidence and its limits

- **Source of the mapping.** Thomas James Wood, who ran the original experiments, confirmed the identifiers, the training and evaluation folds, and that the `best_epoch_f1_fold*.pth` files were not used. The local filenames and their fold directions are also documented in the original notebook (`original_experiments/`).
- **Not independently verified.** The mapping has not been checked against file checksums, download records or training logs. No publisher-supplied checksum for the LITHOS checkpoints was found.
- **Open discrepancy.** An inspection of the public LITHOS model folder on 24 October 2026 found `best_epoch_f1_fold1.pth` and `best_epoch_f1_fold2.pth` (2.51 GB each) under `LITHOS_Baseline/LITHOSBaseline_DualPolarization/Multiclass`. That location suggests dual-polarisation models, whereas the confirmation above describes these files as single-ViT models. Files named `LITHOS_BASELINE_FOLD{1,2}_25_CLASSES` were not seen in that listing. The public location of checkpoints A and B therefore still needs to be documented before anyone tries to regenerate the predictions.
- **Not regenerated.** The predictions and perturbation results in `data/saved_records/` have not been regenerated from the images and checkpoints.
- **Consistency evidence.** The saved predictions reproduce the published LITHOS two-fold mean metrics after rounding: accuracy 0.623, macro recall 0.415, macro precision 0.519 and macro-F1 0.432. This agreement supports the checkpoint attribution but does not prove it.

## Model details recorded in the original experiments

- **Architecture:** `polar_vit_arch.py` at LITHOS commit `ac3391035ea86b9e5d744d0974a061052d033602`. Two frozen ImageNet-pretrained ViT-L/16 encoders are joined by cross-attention decoder layers.
- **Labels:** 25 categories. Raw labels 4, 21, 25 and 26 are merged into Other, following `data_loader.py`.
- **Preprocessing:** `torchvision` `ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1.transforms()`, which gives 224 × 224 inputs.
- **Software:** PyTorch 2.5.0 on CUDA, seed 0. The scikit-image and pytorch-grad-cam versions were not recorded.
