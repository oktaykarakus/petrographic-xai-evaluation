# Original experiment code

`pipelineLithosXAI_code_only.ipynb` is a copy of the notebook that ran the original experiments: whole-input replacement, the five attribution procedures, superpixel deletion with random reference orders, and the qualitative examples. It was written by Thomas James Wood for his MSc dissertation (Cardiff University, 2026).

- **Source:** `code/pipeline/pipelineLithosXAI.ipynb`, SHA-256 `e4fe8b99d36f82dbe3d2db5e66be001d8ba3f3ecae39b5e3255a672ff0437b3e`.
- **Changes:** all cell outputs and execution counts were removed, because the outputs embed LITHOS thin-section imagery. A note was added to the notebook metadata. Cell sources are unchanged.
- **Not executed for this repository.** To run it you need the LITHOS dataset, the checkpoints, the pinned LITHOS code and a GPU (see `../docs/reproducibility.md`). Its cells also contain a parameter-randomisation sanity check and a latent-space analysis, which are not reported in the manuscript.
