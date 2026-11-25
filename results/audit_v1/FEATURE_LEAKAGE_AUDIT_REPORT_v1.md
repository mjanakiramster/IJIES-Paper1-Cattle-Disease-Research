# Feature-leakage audit: overall report (v1, 2026-11-21)

**Scope.** Audit stages A1–A10 plus block-size sensitivity, run on development folds only. The confirmatory partition was not used. The confirmatory result is **not** reconfirmed by this audit; it stands as the evidence observed once on 2026-11-20. Every number below comes from an executed script; the file paths are listed at the end.

## 1. Results by stage

| Stage | What was done | Result |
|---|---|---|
| **A1** Provenance | 74 features traced to code (42 new, 22 old-code rates/density, 10 excluded) | 31 Passed by executed checks; 43 label-derived left for A2–A4; 0 Failed |
| **A2** Reference implementation | Separate loop-based code from the definitions; 20 splits (15 outer + 5 inner), plus a non-default configuration; 1.7 M values (new code), 1.1 M (old code) | Kernel, density, IDW and kNN-distance features match to ≤ 1.8×10⁻¹⁴ (0 mismatches). kNN *rates* differ in 1,001 of 570,915 values: **every case (64 new, 50 old) is an exact distance tie at the k-th neighbour**. Coordinates have 0.1° precision, so 93% of locations have tied neighbours, and the code and the reference order exact ties differently (distance difference ≤ 10⁻¹² km). This is a definition ambiguity, not leakage. |
| **A3** Label perturbation | Flip all test-fold labels (15 splits); flip 1 training household (20); flip 1 disease of 1 training household (20); prior check | Test flips change **0** features (new and old code), **0** priors, and **0** of 46,437 library predictions; the FS5 ExtraTrees predictions are bit-identical. A training flip never changes the household's own features or its block's features (0/20), while rows outside the block do change (as intended). Single-disease flips change only that disease's columns (20/20); density terms are never affected. Old code: the household's own features change (40/40, at most 0.0003), which is the known prior issue (A5). |
| **A4** Negative controls | Full pipeline rebuilt on permuted labels: 50 household-level + 50 village-level permutations (reduced from 100 for compute; p resolution 0.02), development repeat 1 | Null ROC-AUC 0.497–0.515; null FS5 − FS1 gain −0.008 to +0.002. **FMD observed gain +0.081 exceeds all 50 permutations in both schemes (p = 0.020).** LSD +0.045: p = 0.020 (household), 0.18 (village). CBPP +0.017: p = 0.14 / 0.24. |
| **A5** Old fallback prior | Corrected prior; 15 affected models refitted (outer and inner folds) on 15 splits; compared with cache (cache reproduces the code to 1.2×10⁻⁷) | Rank average (a): CBPP +0.0007, FMD +0.0004, LSD −0.0027 (p = 0.16). Best single model unchanged for every disease. Single-model ranking Kendall τ 0.87–0.91; the largest single-model AUC change is 0.016. Boosted-tree predictions moved (Spearman ≥ 0.78) but discrimination barely changed. **Not consequential**; the Stage 6 gate decision is unchanged. |
| **A6** Proximity | Training locations within 25/50/100 km of any test location removed, plus size-matched random removal | FMD gain: +0.102 (none removed), **+0.125 / +0.169 / +0.062** (25/50/100 km buffers; 15/15, 15/15, 11/15 folds positive); matched removal +0.078 / +0.089 / +0.087. Absolute FS5 AUC falls with distance (0.665 → 0.628 → 0.628 → 0.559). LSD gain vanishes at 50–100 km and under matched removal. CBPP shows no gain anywhere. |
| **Block size** | New label-free partitions with 120 and 160 km blocks (3 × 5) | FMD gain +0.118 / +0.100 (12/15, 15/15). Absolute AUCs drop (FMD FS5 0.665 → 0.622 / 0.626; CBPP 0.697 → 0.668 / 0.664). The 81 km size came from a variogram of all labels; a development-only variogram is not possible because the same households are in every partition. |
| **A7** Temporal | Share of later-survey information; year-restricted features; forward temporal validation; within-year AUC | On average 31% of the 150 km kernel weight (23% of the 15 nearest households) comes from **later** surveys (2012 households: 83%). **Forward validation (train 2012–2015, test 2018–2019): ROC-AUC 0.35–0.43 for all diseases and both feature sets**, so the models do not transfer forward in time. Year-restricted features (Scenario B, other-years) gave higher AUCs (0.74–0.78), but these are **artefacts**: the fit population changes with the query's survey year, so the features encode the year. |
| **A7d** Survey-year confounding | Prevalence by year; within-year ROC-AUC | Prevalence differs greatly by survey year (CBPP 38–43% in 2012–2015 vs 0.5% in 2019; LSD 7.5% in 2018 vs 61% in 2019). **Within-year** ROC-AUC is much lower: FS1 0.58/0.53/0.54 and FS5 0.56/0.59/0.49 (CBPP/FMD/LSD). **The FMD gain survives within years (+0.062, 12/15 folds)**; CBPP (−0.015) and LSD (−0.046) do not. |
| **A8** Row order and integrity | Rows shuffled and features rebuilt for both codes; ID, location, coordinate and block checks | Maximum difference after ID matching 0.0 (both codes). IDs unique; each location has one coordinate pair and one block per partition; no two locations share coordinates; households at a location get identical features; 13 multi-year locations get one pooled value. |
| **A9** Train/test distributions | 15 folds, all 24 rate features plus 7 information-condition measures | Standardised train–test differences ≤ 0.14 for every feature. Prior-dominated (50 km fallback) rows: 1.3% train vs 2.4% test; nearest contributing location 41 vs 43 km; 31 vs 32 contributing locations within 150 km. Training and test features are built under the same information conditions. |
| **A10** Pipeline isolation | 15-item checklist and pipeline diagram | Every fitted step (scaling, rate features, calibration, threshold, model selection, ECDF ranks, ensemble weights) uses training or inner-OOF data only. One deviation: the old-code prior (A5). |

## 2. Feature classification

| Feature group | Classification |
|---|---|
| 5 environmental variables, 6 climate indices, 4 coordinate terms | **Verified leakage-free** (no labels; A1 permutation check; A8) |
| Sampling-density terms (kernel density, kNN distance) | **Verified leakage-free** (label-free, A1/A3). They reflect survey design. |
| Own-disease neighbourhood rates (kernel, kNN; new code) | **Verified free of direct target leakage** (A2, A3, A4, A8). **Legitimate but dependent on surveillance data from other areas.** **Temporally pooled**, so valid for retrospective spatial prediction (Scenario A), **not** for prospective prediction (Scenario B). |
| Cross-disease rates (new code) | Same as above. A3 confirms each column uses only its own disease's labels of other locations. |
| Old-code rates (23-model library) | **Minor deviation from the stated rule** (prior includes the own block for training rows; never test labels). **Not consequential** (A5). |
| kNN rates (both codes) | Tie-break among exactly equidistant neighbours is implementation-dependent (A2b). This is a reproducibility note, not leakage. Recommendation: break ties by location ID after rounding distances to 1 m. |

## 3. Final decisions

| # | Question | Decision |
|---|---|---|
| 1 | Neighbourhood features leakage-free? | **Pass with limitations**: no direct target leakage; Scenario A only. |
| 2 | Cross-disease features leakage-free? | **Pass with limitations**: same. |
| 3 | Climate indices leakage-free? | **Pass** |
| 4 | Temporal information leakage? | **Pass with limitations for retrospective mapping** (31% of neighbourhood weight is from later surveys). **Fail for any prospective / forecasting claim**: forward AUC 0.35–0.43. |
| 5 | Old fallback issue consequential? | **Pass**: rank-average change ≤ 0.003, rankings and best models unchanged. |
| 6 | FMD +0.068 supported by the development audit? | **Pass with limitations**: the development gain (+0.10) beats every permutation and survives 25–100 km buffers, 120/160 km blocks and the within-year analysis (+0.06). Absolute within-year discrimination is low (0.59). |
| 7 | Depends on nearby surveillance? | **Pass with limitations**: the gain persists without data within 50 km and shrinks beyond 100 km; absolute AUC falls with distance. |
| 8 | Sensitive to block size? | **Pass with limitations**: absolute AUCs drop by 0.02–0.04 with larger blocks; the FMD gain is stable. |
| 9 | Experiments to rerun | No rerun needed for leakage. **Must add:** within-year (survey-campaign) ROC-AUC alongside every main result; the forward temporal result as a limitation; corrected-prior rank-average values (development) in the supplement. Optional: deterministic kNN tie-break. |
| 10 | What Paper 1 can report | See Section 4. |

## 4. What Paper 1 can responsibly report

- `Weight` leakage; spatial vs random validation gap; the locked-partition results as observed once (2026-11-20).
- The **FMD engineered-feature gain**: confirmed on the locked partition, passing the leakage audit, and robust to proximity, block size and survey year.
- LSD and CBPP: no robust feature gain.
- Negative optimisation results (Stages 5–6, 5c–5d).
- **New and essential limitations:**
  - Outcome prevalence differs strongly between survey campaigns (years), and campaigns covered different areas. Much of the overall spatial ROC-AUC therefore reflects differences *between campaigns*. Within-campaign discrimination is modest: about 0.53–0.59.
  - The models are for **retrospective mapping of reported occurrence**, not forecasting: forward temporal validation fails.
  - Outcomes are household-reported occurrence.
- Do **not** claim prospective prediction, laboratory-confirmed disease, or a new independent confirmation.

## Files (all under `results/audit_v1/`)

- **A1/** provenance table, label uses, summary
- **A2_A3_A8/**
  - `A2_reference_comparison_*`, `A2b_knn_tie_*`, `A2_old_code_vs_correct_definition_by_feature_v1.csv`
  - `A3_label_perturbation_v1.csv`
  - `A8_row_order_v1.csv`, `A8_integrity_v1.json`
- **A4/** `A4_all_runs_v2.csv`, `A4_null_summary_v2.csv`, `A4_negative_controls_v2.png` (v1 script stopped at 30/29 permutations; v2 completes 50 each, reusing them)
- **A5/** fold metrics, rank-average summary, ranking change, prediction changes, meta
- **A6/** fold results, summary, figure (proximity and block size)
- **A7/**
  - later-survey share
  - Scenario B and other-years results (artefacts; kept for the record)
  - `A7b_forward_temporal_validation_v1.csv`
  - `A7d_prevalence_by_survey_year_v1.csv`, `A7d_within_year_*`
- **A9/** fold distributions, train-vs-test summary
- **A10/** checklist, `A10_pipeline_diagram_v2.png` (v1 had overflowing labels)

**Scripts:** `src/audit_*.py` and `src/audit_common_v1.py`, which contains the independent reference implementation.
