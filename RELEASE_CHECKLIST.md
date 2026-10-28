# Release checklist

**Current state: local staging only.** No Git repository has been initialised, nothing has been pushed, and no remote, release, GitHub Pages site, archive deposit or DOI exists.

## Stage 1: private GitHub repository (only after Oktay Karakuş approves)

- [ ] Oktay Karakuş approves in writing the creation of a **private** repository and the account or organisation that will host it.
- [ ] Create the repository with private visibility set at creation, for example `gh repo create <owner>/<name> --private`. Do not create it public and switch it afterwards.
- [ ] Add no collaborators automatically. Each invitation needs the corresponding author's approval.
- [ ] Leave Releases, GitHub Pages, Discussions and Wiki unused.
- [ ] Run the pre-push checks below, then push once.

## Pre-push checks (repeat before every push)

- [ ] `python analysis/scripts/analyze.py && python analysis/scripts/verify.py && python scripts/render_figures.py && python scripts/make_supplementary_tables.py && python scripts/check_manuscript_numbers.py` all succeed, and `git status` shows no unexpected changes.
- [ ] `shasum -a 256 -c MANIFEST.sha256` passes. Regenerate the manifest only after a deliberate, documented change.
- [ ] Staged files contain no LITHOS images, Figure 3 image or model weights (`*.pth`, `*.pt`, `*.ckpt`, `*.safetensors`).
- [ ] No credentials, tokens, personal e-mail addresses or absolute local paths.
- [ ] No file larger than 10 MB.

## Stage 2: public release (separate task, after the paper is accepted)

- [ ] Paper accepted. Add the journal, year, volume, article number and DOI to `CITATION.cff` and `README.md`.
- [ ] Licence choice confirmed by all authors, and Cardiff University if its IP policy applies. Add `LICENSE` file(s): one for the original code, and one for the derived data and figures, which must stay compatible with CC BY-NC-SA 4.0.
- [ ] Redistribution of the LITHOS-derived records in `data/saved_records/` is confirmed to comply with CC BY-NC-SA 4.0, with attribution, NonCommercial and ShareAlike terms stated in `data/README.md`.
- [ ] Decide whether `original_experiments/` is released. Confirm Thomas James Wood's agreement and any third-party code attribution, for example MIT notices for LITHOS-derived snippets.
- [ ] Figure 3 image: exclude it unless its redistribution under the LITHOS licence has been confirmed. The provenance record stays.
- [ ] Public location of checkpoints A and B documented, or the open discrepancy in `docs/checkpoint_provenance.md` kept as stated.
- [ ] Corresponding-author contact address added to `README.md`.
- [ ] Data and Code availability statements in the article updated with the final repository URL.
- [ ] Visibility changed to public only with the corresponding author's explicit approval. Creating a release or archive DOI (for example via Zenodo) is a separate decision.
