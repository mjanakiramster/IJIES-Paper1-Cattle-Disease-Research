# IJIES Paper 1, IJIES version 5 (1,000-permutation negative controls) - README rev5 v1

Spatial generalisation and survey-campaign confounding in machine-learning models for retrospective mapping of cattle disease occurrence in Ethiopia.

## What changed from rev4
- Permutation negative controls extended from 100 to 1,000 permutations per scheme (household-level and village-level), development repeat 1, with exactly the same code and seed rule. The 900 added permutations per scheme are in `results/paper1_rev5_v1/E5_permutations_ext_v1/`; the original 100 and their tables are unchanged.
- New table `results/paper1_rev5_v1/tables/Table10_negative_controls_1000_v1.csv` (manuscript Table 4, Supplementary Table S8) and figure `results/paper1_rev5_v1/figures/Fig5_permutation_null_1000_v2` (Supplementary Figure S6; script `fig5_permutation_v5.py`).
- Manuscript IJIES v5: simplified abstract, 1,000-permutation results, a sentence on the Gaussian-process (kriging) spatial baseline, practical guidance in the Conclusion. Supplement IJIES v2: Table S8 and Figure S6 updated.
- Notebook v6 (`IJIES_Paper1_Experiments_22sections_executed_v6.ipynb`): computes the 900 added permutations per scheme if missing, checks the 1,000-permutation table against the stored file, and draws Figure S6 (figure script v5).

## Contents of the code package
| Path | What it is |
|---|---|
| `IJIES_Paper1_Experiments_22sections_executed_v6.ipynb` | Final notebook, executed (current) |
| `IJIES_Paper1_Experiments_22sections_unexecuted_v6.ipynb` | The same notebook without outputs |
| `src/` | All analysis code (`src/paper1/`), including `rev5_perm_extension_v1.py`, `rev5_perm_tables_v1.py`, `fig5_permutation_v5.py` |
| `configs/` | Frozen configuration and decision logs |
| `data/reference/` | Fold file and reference data; put the three raw data files in `data/raw/` |
| `results/` | All checkpoints and result tables, including `paper1_rev5_v1/` |
| `manuscript/` | Builders for the manuscript and supplement (`build_manuscript_ijies_v5.py`, `build_supplement_ijies_v2.py`) and the renderer |

## Reproduce
1. `pip install -r requirements.txt`
2. Put `cbpp.xlsx`, `fmd.xlsx` and `lsd.xlsx` in `data/raw/`.
3. Run the notebook v6 from the project root. Existing checkpoints are loaded and every table is checked, which takes a few minutes. Missing checkpoints are computed: measured on a 2-core 2.1 GHz machine, the original experiments take about 1 h 45 min as one process, and the 1,800 added permutations about 7.5 h as one process (about 4 h with one worker per scheme: `PYTHONPATH=src python -m paper1.rev5_perm_extension_v1 run household 100 999` and the same with `village`). Each checkpoint is one file, so an interrupted run continues where it stopped.
4. Build the paper: `python manuscript/build_manuscript_ijies_v5.py`, `python manuscript/build_supplement_ijies_v2.py`, then `node manuscript/render_docx_v2.js <spec>.json <out>.docx`.

## Scope
Retrospective mapping of household-reported occurrence. Not forecasting, not laboratory-confirmed, not causal. `Weight`, survey year, region, the year-region code and identifiers are never predictors (one library model uses year-region group sizes as training weights; disclosed). No optimisation or swarm algorithm is used in any analysis of the paper; the completed pre-registered optimisation study is reported in Supplementary Section S2 only.
