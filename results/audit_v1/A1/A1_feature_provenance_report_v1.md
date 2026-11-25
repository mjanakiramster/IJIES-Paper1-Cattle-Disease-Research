# Feature-leakage audit, A1: feature provenance (v1, 2026-11-21)

**Scope:** every feature any model in the study can receive. That covers the new pipeline (the FS5 features, plus the IDW rates, which exist only in the optimisers' search space) and the old run-3 pipeline (feature set ELK, which feeds the 23-model rank average). Feature names were taken from the executed code, not from documentation. Nothing was fitted, and only development data were used.

**Files**
- `A1_feature_provenance_v1.csv`: 74 rows, with all columns required by the audit prompt
- `A1_label_uses_outside_features_v1.csv`: places where labels enter outside the features
- `A1_summary_v1.json`
- `A1_old_vs_new_label_free_check_v1.json`
- Script: `src/audit_A1_feature_provenance_v1.py`

## Inventory

| Group | New pipeline | Old pipeline | Labels used? |
|---|---|---|---|
| Original environmental variables | 5 | 5 (same values) | No |
| Climate indices | 6 | 5 (same values; Precip_x_RH is new only) | No |
| Coordinate terms | 4 | 4 (same values) | No |
| Own-disease and cross-disease rates (kernel 50/150/400 km, kNN 5/15/40) | 18 (+3 IDW, search space only) | 18 | **Yes**, training households' labels |
| Sampling-density terms | 6 | 4 | No (household counts) |
| Excluded variables (Weight ×3, Year, Region, Year_region, IDs, folds) | 0 | 0 | — |

## Executed checks in A1

1. **Label-free features (21): Passed.** With all three disease labels permuted, every climate, coordinate and density feature stayed bit-for-bit identical. The old pipeline's 14 shared label-free columns equal the new ones exactly (maximum difference 0.0), so this applies to both pipelines.
2. **Excluded variables: Passed.** No Weight, Year, Region, Year_region, ID or fold column appears in any feature list of either pipeline.

## Code trace of the label-derived rates (to be verified empirically in A2–A4)

**Source rows.** Counts and positives come from `Y[fit_idx]` only (features.py L46–L47; run3 L165–L166 and L183–L184), so test-fold labels never enter by construction.

**Spatial exclusion.**
- Every query location, training or test, excludes all fit locations in its own 81 km block.
- A training household therefore never uses its own label or any label from its own block.
- A test location's own block is a test block, so the rule removes nothing extra for it.
- A household's own other-disease status is never used, because the cross-disease rates also come only from other locations' fit rows.

**Prior.**
- New code: p0 excludes the query's own block.
- Old code: p0 is the rate over all fit rows, which for a training query includes its own block and its own household (weight about 1/2,670). **Known issue, A5.** It cannot involve test labels.

**Temporal.**
- All survey years are pooled: 2012 (224 households), 2014 (684), 2015 (656), 2018 (1,341) and 2019 (435). Only 13 of 462 locations were surveyed in more than one year.
- A 2012 household's rates therefore use 2014–2019 reports. That is valid for retrospective spatial prediction (Scenario A) but not for prospective prediction (Scenario B). **To be quantified in A7.**

## Label uses outside the features (for A10)

- GP-kriging trains on training-location empirical logits.
- CatBoost-groupbalanced uses Year_region **only as training sample weights**.
- The village-level CatBoost trains on training-location means.
- The ECDF rank scaling uses training out-of-fold scores.
- The threshold is the Youden index on inner out-of-fold predictions.

All of these are training-only by code trace, but are not yet verified.

## Status after A1

- 31 rows **Passed** by an executed check: 21 label-free features and 10 excluded variables.
- 43 rows **Unresolved**: 39 label-derived rates and 4 old-code density terms, pending A2 (reference implementation), A3 (perturbation) and A4 (negative controls).
- **0 rows Failed.**

**Likely reviewer criticisms**
- "Neighbourhood rates are a form of target encoding": A3 and A4 address this.
- "Pooling years uses future reports": A7.
- "Gains come from nearby surveyed villages, not transferable knowledge": A6.
- "The block size was chosen from all labels": block-size sensitivity.
