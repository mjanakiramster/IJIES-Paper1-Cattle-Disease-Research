"""Supplementary Material for the shortened IJIES version, builder v2: Table S8 and Figure S6 with 1,000 permutations per scheme
(rev5 table and figure); executed notebook v6 named. Builder v1: rev4 v2 supplement plus the tables and figures that left the
main text (Tables S24-S28: feature sets, models, calibration, paired differences, campaign-specific models; Figures S10-S15).
Earlier: Supplementary Material, revision 4 builder v2 (reference table v4). Builder v1 (final correction round): reference table v3; notebook v5 named; Table S7 note states
that the hypotheses were fixed after the development results and gives the fold-level test; Table S8 note gives the p-value resolution
and the Holm family. Earlier, revision 3 builder v3: Table S18 note defines the error measure and
the per-repeat counts. Builder v2: group-balanced weight formula stated exactly
(proportional to 1/n_g, rescaled to mean 1, as in the library code). Builder v1 notes: new in rev3 v1: method notes on the group-balanced weights,
SVM decision scores, condition-specific test/train ratio, SHAP, partial dependence, area of applicability and survey-year identification;
Tables S15-S23 (ablation, SHAP, AOA, metrics with CIs, campaign identification, ensemble without the group-balanced model,
train-test distances); Figures S5-S9 (paired folds, permutation nulls, calibration bands, partial dependence, distances).
Run from the project root: python manuscript/build_supplement_rev3_v1.py
Earlier (rev v2): Supplementary Material for the REVISED IJIES Paper 1 (optimisation study moved here as Section S2; new Tables S4, S12-S13; optimisation results in Table S14).
Run from the project root after the notebook has been executed: python manuscript/build_supplement_rev_v2.py
rev v2: Figure S2 v2 (calibration labels no longer overlap).
Based on build_supplement_v2.py:
v2: two-decimal threshold metrics, readable condition names in Tables S5-S6, Figure S4 v2."""
import json, re
from pathlib import Path
import pandas as pd
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'results'; P1 = R / 'paper1_v1'; TAB = P1 / 'tables'; FIG = P1 / 'figures'
REV = R / 'paper1_rev_v1'; DC = REV / 'design_checks'; R2 = R / 'paper1_rev2_v1'; T2D = R2 / 'tables'; F2D = R2 / 'figures'; RFG = REV / 'figures'
DIS = ['CBPP', 'FMD', 'LSD']
B = []
def add(t, text=None, **kw): B.append({'type': t, **({'text': text} if text is not None else {}), **kw})
def tab(caption, header, rows, widths, note=None, size=14):
    B.append({'type': 'caption', 'text': caption, 'keepNext': True})
    B.append({'type': 'table', 'header': header, 'rows': [[str(c) for c in r] for r in rows], 'widths': widths, 'note': note, 'size': size})
def fig(name, caption, width=620, folder=None):
    p = (folder or FIG) / f'{name}.png'; w, h = Image.open(p).size
    B.append({'type': 'figure', 'path': str(p), 'aspect': h / w, 'alt': caption.split('**')[-1][:80], 'width_px': width}); B.append({'type': 'caption', 'text': caption})
f3 = lambda v: f'{float(v):.3f}'

add('title', 'Supplementary Material')
add('center', 'Spatial Generalisation and Survey-Campaign Confounding in Machine-Learning Models for Retrospective Mapping of Cattle Disease Occurrence in Ethiopia', size=22)
add('center', 'Sudhirvarma Sagiraju and [co-authors to be confirmed]', size=20)
add('para', 'This file gives method details, the pre-registration history, full result tables and additional figures. All values were produced by the executed notebook IJIES_Paper1_Experiments_22sections_executed_v6 (22 sections plus an appendix) with the frozen configuration configs/ijies_paper1_config_v1.yaml. Development results use the corrected code (deterministic k-nearest-location ties and the own-block prior correction). Results labelled "as observed" (the confirmatory test and the optimisation study) were produced earlier with the pre-correction code and are shown unchanged.')

# ------------------------------------------------------------------ S1 methods
add('h1', 'S1. Method details')
add('h2', 'S1.1 Neighbourhood features')
add('para', 'For each split, rates are fitted on the training households only. For a query location in 81 km block g, all training locations in block g are removed before the rate is computed. This applies to training rows and to test rows in the same way. The prior p_{0,d} is the training rate of disease d outside block g, and the prior weight is c = 1 household.')
add('para', 'The k-nearest-location rate sums positives and households over the k nearest training locations outside the block, adds the prior, and divides. Because coordinates are recorded to 0.1 degree, 93% of locations have at least one exact distance tie. Neighbours are therefore sorted by distance rounded to 10^{-6} km and then by location ID, so the result does not depend on the order of the input rows.')
add('para', 'The density terms are log kernel densities of training households (50, 150 and 400 km) and log distances to the k-th nearest training location (k = 5, 15, 40). They use household counts only, not outcomes.')
add('h2', 'S1.2 Ensemble, calibration and thresholds')
add('para', 'Each of the 23 library models produces a test score and an inner out-of-fold (OOF) score for the training households, using five inner spatial folds. The test score is converted to its empirical cumulative distribution value within the model\'s OOF scores, and the 23 values are averaged with equal weight. Comparator (b) is the single model with the highest inner OOF ROC-AUC.')
add('para', 'All library outputs are ranked before averaging, so outputs on different scales can be combined. The support vector machine contributes its decision-function values (no probability step inside the library), and the village-level CatBoost model is a location-level regressor of the mean outcome (depth 4), whose predictions are assigned to the households of each location. The group-balanced CatBoost model weights each training household in proportion to 1 / n_{g}, where n_{g} is the number of training households in its survey year-region group, and rescales the weights to a mean of 1 (in the audited split: 15 groups, weights 0.43 to 88.9). The year-region code is used only for these weights and never as a predictor; the weights use training rows only and do not depend on outcomes.')
add('para', 'Platt calibration is a logistic regression of the inner OOF labels on the inner OOF score, applied to the test score. When the fitted slope is negative, the calibrated probability is set to the training prevalence (the "Platt fallback" count in Table S1). The classification threshold maximises the Youden index on the calibrated inner OOF probabilities.')
add('h2', 'S1.3 Statistical inference')
add('para', 'For paired differences over the 15 development folds, the corrected resampled t statistic uses the variance inflation (1/k + n_{test}/n_{train}) with k = 15 and 14 degrees of freedom {Nadeau2003}. For the 81 km design, n_{test}/n_{train} = 668/2672. For the proximity-buffer and block-size conditions (Tables S5 and S6), the mean numbers of test and training households of each condition were used, because removing training locations or enlarging blocks changes this ratio. The location-cluster bootstrap resamples the 462 locations with replacement (1,000 resamples, seed 20261122). Within each repeat, out-of-fold test scores from the five folds are pooled after conversion to quantiles of the corresponding training OOF distribution, so that scores from different folds are on a common scale. Holm correction is applied across the three diseases.')
add('h2', 'S1.4 Negative controls')
add('para', 'Household-level permutation shuffles the three outcome vectors jointly across all households. Village-level permutation shuffles the outcome vectors of whole locations between locations in the same household-count bin, so that the within-village clustering of outcomes is kept. In both schemes, the neighbourhood features, the ExtraTrees models and the evaluation are rebuilt from the permuted labels on development repeat 1. The permutation p-value is (1 + number of null gains at least as large as the observed gain) / (1 + number of permutations).')
add('h2', 'S1.5 Feature ablation and explainability')
add('para', 'The ablation fitted ExtraTrees (300 trees, minimum leaf size 10, square-root feature sampling) with the nested sets FS1 to FS5 on the same 15 development splits as the main analysis; FS1 and FS5 are the same runs as in the main tables. Each step was tested with the corrected resampled t-test and the location-cluster bootstrap, and Holm correction was applied across the 12 steps (four steps × three diseases).')
add('para', 'SHAP values were computed with the exact TreeExplainer algorithm {Lundberg2020} for the FS5 ExtraTrees model fitted in each fold of development repeat 1, on the probability scale of the positive class, for the test households of that fold {Lundberg2017}. Partial dependence {Friedman2001} was computed on the training households of each fold (20 grid points between the 5th and 95th percentiles), so no test labels were used. Grouped permutation importance is the drop in test ROC-AUC when all features of a group are permuted together (15 folds).')
add('h2', 'S1.6 Area of applicability')
add('para', 'For each fold, the predictors were standardised with the training means and standard deviations and multiplied by the impurity importances of the fitted ExtraTrees model. The dissimilarity index (DI) of a household is its Euclidean distance to the nearest training household in this space, divided by the mean distance between training households {Meyer2021}. Training DIs were computed across the five inner spatial folds (distance from each inner validation household to the inner fitting households). The AOA threshold is the upper whisker of the training DIs (Q3 + 1.5 IQR, capped at the maximum). Test households with DI above the threshold are outside the AOA. Only surveyed locations were evaluated.')
add('h2', 'S1.7 Survey-year identification')
add('para', 'An ExtraTrees classifier (300 trees, minimum leaf size 10) predicted the survey year of each test household from FS1, from the 18 neighbourhood rates alone, or from FS5, under the same spatial splits and with the same leakage-safe feature construction. For each survey year present in a test fold, the one-versus-rest ROC-AUC was computed; the macro ROC-AUC is their mean. Intervals use the corrected resampled t-test over the 15 folds.')

S5S = json.load(open(R / 'stage5_v1' / 'optimiser_statistics.json')); S5R = S5S['friedman']['best_inner_fitness']['mean_ranks']; S5P = S5S['friedman']['outer_ROC_AUC']['friedman_p']
_S5d = pd.read_csv(R / 'stage5d_noise_v1' / 'fitness_variant_summary.csv'); S5SP = _S5d[(_S5d.variant == 'V0') & (_S5d.subset == '21 optimiser best only')].mean_spearman.iloc[0]
G6 = json.load(open(R / 'stage6_v1' / 'go_no_go_gate.json'))
add('h1', 'S2. Pre-registration history and the optimisation study')
add('para', 'This section reports a completed study that is outside the scope of the revised paper. It is given so that every pre-registered analysis is visible; none of its results are used in the paper.')
add('para', 'The project was pre-registered in Stage 2, before any new model was fitted (configs/experiment_config.yaml and configs/decision_log.json). The registered aim was to test whether a spatially robust variant of social group optimisation (SR-SGO) {Satapathy2016} {Naik2020}, used to choose feature groups, bandwidths and ensemble weights, would beat the rank average (a) and the inner-selected model (b) on the sealed confirmatory partition. A go/no-go gate on the development folds was also registered: proceed to the confirmatory test only if SR-SGO had a higher mean ROC-AUC than (a) and a positive pooled bootstrap difference for at least two of the three diseases.')
add('para', f'Before SR-SGO, seven optimisers were compared on development repeat 1: SGO, modified SGO (MSGO), particle swarm optimisation {{Kennedy1995}}, a genetic algorithm {{Katoch2021}}, differential evolution {{Storn1997}}, the grey wolf optimiser {{Mirjalili2014}} and random search. Each searched a 24-variable space with 10 agents and 150 evaluations of an inner spatial cross-validation fitness (mean minus 0.5 SD of ROC-AUC), giving 315 runs (Table S14). The optimisers differed in the inner fitness they reached (Friedman p < 0.001; mean ranks, lower is better: PSO {S5R["PSO"]:.2f}, DE {S5R["DE"]:.2f}, GWO {S5R["GWO"]:.2f}, SGO {S5R["SGO"]:.2f}, MSGO {S5R["MSGO"]:.2f}, random search {S5R["RANDOM"]:.2f}, GA {S5R["GA"]:.2f}) but not in outer ROC-AUC (Friedman p = {S5P:.2f}). Among the best configurations, the Spearman correlation between inner fitness and outer ROC-AUC was {S5SP:+.2f}, so the search target did not track real spatial performance.')
add('para', f'SR-SGO then failed the registered gate for all three diseases ({G6["diseases_passing"]} of 3) and was not evaluated on the confirmatory partition. On 2026-11-20, before the partition was opened, it was re-allocated once to two new hypotheses: H1, FS5 has a higher ROC-AUC than FS1 for both FMD and LSD (confirmed only if both bootstrap intervals were above 0); and H2, a two-sided comparison of FS5 with (a). These hypotheses were written after the development results were known, and the registration recorded the development estimates. The partition was then opened once (Table S7). The optimisation study and the confirmatory test both used the code before the two corrections described in the paper.')

order = []   # IEEE-numbered references for the supplement
REFS = pd.read_csv(P1 / 'references' / 'reference_verification_table_v4.csv').set_index('key')

# ------------------------------------------------------------------ tables
add('h1', 'S3. Supplementary tables')
T4 = pd.read_csv(TAB / 'Table4_overall_spatial_performance_v1.csv')
tab('**Table S1.** Full development performance (15 spatial splits, Youden threshold chosen on inner out-of-fold data)',
    ['Disease', 'Model', 'ROC-AUC fold mean (SD)', 'PR-AUC (no-skill)', 'Brier', 'Cal. intercept', 'Cal. slope', 'Platt fallback', 'Sens.', 'Spec.', 'Bal. acc.', 'F1', 'MCC'],
    [[r.Disease, r.Model, r['ROC-AUC fold mean (SD)'], r['PR-AUC (no-skill)'], f'{r.Brier:.3f}', f"{r['Cal. intercept (median)']:.2f}", f"{r['Cal. slope (median)']:.2f}", r['Platt fallback folds']] + [f'{v:.2f}' for v in (r.Sensitivity, r.Specificity, r['Balanced acc.'], r.F1, r.MCC)] for _, r in T4.iterrows()],
    [700, 1500, 1150, 1050, 600, 750, 650, 750, 550, 550, 650, 500, 550], note='Calibration intercept and slope are medians over folds. Slopes above 1 mean the predicted probabilities are too close to the average risk.')
YA = pd.read_csv(TAB / 'TableS_year_specific_AUC_v1.csv'); years = sorted(YA.year.unique())
cnt = YA.drop_duplicates(['disease', 'year']).set_index(['disease', 'year'])
rows = []
for d in DIS:
    for m in ['FS1', 'FS5', '(a) Rank average', '(b) Inner-selected single']:
        q = YA[(YA.disease == d) & (YA.model == m)].set_index('year').AUC
        rows.append([d, m] + [f3(q[y]) if y in q.index and pd.notna(q[y]) else 'n/a' for y in years])
    rows.append([d, 'Households (positives)'] + [f'{int(cnt.loc[(d, y), "n"])} ({int(cnt.loc[(d, y), "positives"])})' for y in years])
tab('**Table S2.** Year-specific ROC-AUC (pooled out-of-fold predictions, mean over the three development repeats)', ['Disease', 'Model'] + [str(y) for y in years], rows,
    [800, 2400] + [1300] * len(years), note='Estimates from years with very few positive households (for example, 2 CBPP cases in 2019) are unstable.')
T11 = pd.read_csv(TAB / 'Table11_ensemble_prior_sensitivity_v1.csv')
tab('**Table S3.** Effect of the two code corrections on the rank-average ensemble (development, 15 splits)', list(T11.columns), T11.values.tolist(), [800, 2700, 1400, 1400, 2200, 1200])
LIBT = pd.read_csv(REV / 'tables' / 'RT3_library_models_spatial_v1.csv')
tab('**Table S4.** Spatial ROC-AUC of the 23 library models (development, 15 splits, corrected code)', list(LIBT.columns), LIBT.values.tolist(), [3400, 2100, 2100, 2100],
    note='Feature-set codes: B = 5 environmental variables; E = environmental, climate and coordinate terms; EL = E plus kernel rates and densities; ELK = EL plus k-nearest rates and distance. GP-kriging uses location-level inputs.')
cols = ['Condition', 'Disease', 'Removed households', 'Min test positives', 'Median nearest training location (km)', 'FS1 ROC-AUC', 'FS5 ROC-AUC', 'FS5 - FS1 [95% CI]', 'Folds > 0', 'FS1 PR-AUC', 'FS5 PR-AUC', 'No-skill PR-AUC']
W = [1000, 700, 900, 800, 1000, 750, 750, 1650, 650, 750, 750, 750]
for k, (name, cap) in enumerate([('Table9_proximity_removal_v1', '**Table S5.** Proximity removal and size-matched random removal (development, ExtraTrees, 15 splits)'),
                                  ('Table8_block_size_sensitivity_v1', '**Table S6.** Block-size sensitivity (exploratory; development, ExtraTrees, 15 splits)')]):
    T = pd.read_csv(TAB / f'{name}.csv')
    LAB = {'base81': '81 km blocks', 'blocks120': '120 km blocks', 'blocks160': '160 km blocks', **{f'buffer_{k}': f'Buffer {k} km' for k in (25, 50, 100)}, **{f'matched_{k}': f'Size-matched {k} km' for k in (25, 50, 100)}}
    tab(cap, cols, [[LAB.get(r[c], r[c]) if c == 'Condition' else (r[c] if not isinstance(r[c], float) else f3(r[c])) for c in cols] for _, r in T.iterrows()], W, note='95% CI: corrected resampled t over 15 folds.')
C = pd.read_csv(TAB / 'TableS_confirmatory_as_observed_confirmatory_summary_v1.csv')
tab('**Table S7.** Confirmatory partition (2 × 5 folds), opened once on 2026-11-20 with the pre-correction code; results as originally observed',
    ['Disease', 'FS1', 'FS5', '(a) Rank average', 'H1: FS5 - FS1 pooled [95% CI]', 'H1 folds better', 'H1 corrected-t p', 'H2: FS5 - (a) pooled [95% CI]', 'H2 folds better'],
    [[r.disease, f3(r['FS1 confirmatory']), f3(r['FS5 confirmatory']), f3(r['(a) confirmatory']), f"{r['H1 FS5 - FS1: pooled']:+.3f} [{r['H1 FS5 - FS1: CI_lo']:+.3f}, {r['H1 FS5 - FS1: CI_hi']:+.3f}]",
      r['H1 FS5 - FS1: folds better'], f"{r['H1 FS5 - FS1: p_corrected']:.3f}", f"{r['H2 FS5 - (a): pooled']:+.3f} [{r['H2 FS5 - (a): CI_lo']:+.3f}, {r['H2 FS5 - (a): CI_hi']:+.3f}]", r['H2 FS5 - (a): folds better']] for _, r in C.iterrows()],
    [800, 750, 750, 1000, 1900, 900, 900, 1900, 900], note='ROC-AUC is the mean of the 10 confirmatory folds; pooled differences and intervals are from the location-cluster bootstrap; p is the fold-level corrected resampled t-test. The hypotheses were recorded before the partition was opened but after the development results were known, so this is a held-out check rather than an independent pre-registered confirmation. Per-fold values are provided in the results folder.')
T10 = pd.read_csv(ROOT / 'results/paper1_rev5_v1/tables/Table10_negative_controls_1000_v1.csv', dtype=str)
T10 = T10[['Scheme', 'Disease', 'Permutations', 'Null FS1 ROC-AUC mean [2.5-97.5%]', 'Null FS5 ROC-AUC mean [2.5-97.5%]', 'Null gain mean [2.5-97.5%]', 'Observed gain',
           'Permutation p (resolution)', 'Null FS5 PR-AUC', 'No-skill PR-AUC', 'Holm p (6 tests)']]
tab('**Table S8.** Negative controls in full (development repeat 1; FS1 and FS5 ExtraTrees rebuilt on permuted labels; values are means over the five folds of repeat 1)', list(T10.columns), T10.values.tolist(),
    [900, 650, 800, 1300, 1300, 1300, 700, 1100, 700, 700, 800], note=f'Permutation p = (1 + null gains at least as large as the observed gain) / (1 + {int(T10.Permutations.min()):,}); the smallest attainable value is {1 / (1 + int(T10.Permutations.min())):.4f}. The first 100 permutations per scheme are those of the original analysis (Holm-adjusted FMD p = 0.059 with 100); the other 900 were added with the same code and seed rule. Holm correction across the six tests (three diseases × two schemes).')
I = pd.read_csv(TAB / 'TableS_grouped_permutation_importance_v1.csv')
groups = ['Environmental (5)', 'Climate indices (6)', 'Coordinates (4)', 'Own-disease rates (6)', 'Cross-disease rates (12)', 'Sampling density (6)']
tab('**Table S9.** Grouped permutation importance for FS5 ExtraTrees: mean drop in test ROC-AUC when a feature group is permuted (SD over 15 folds)',
    ['Feature group'] + DIS, [[g] + [f"{I[(I.disease == d) & (I.group == g)].mean_AUC_drop.iloc[0]:+.3f} ({I[(I.disease == d) & (I.group == g)].sd_over_folds.iloc[0]:.3f})" for d in DIS] for g in groups],
    [3200, 2000, 2000, 2000], note='Own-disease rates are the six kernel and k-nearest rates of the modelled disease; cross-disease rates are the twelve rates of the other two diseases.')
FD = pd.read_csv(TAB / 'feature_dictionary_v1.csv')
DESC = {'Altitude': 'Altitude (m)', 'Temperature': 'Mean temperature', 'Precipitation': 'Precipitation', 'Relative_humidity': 'Relative humidity (%)', 'Wind_speed': 'Wind speed',
        'SL_temperature': 'Temperature + 0.0065 x altitude (sea-level equivalent)', 'DeMartonne': 'De Martonne aridity index: precipitation / (temperature + 10)',
        'Temp_x_RH': 'Temperature x relative humidity / 100', 'Precip_x_RH': 'log(1 + precipitation) x relative humidity / 100', 'log_Precip': 'log(1 + precipitation)',
        'Wind_x_RH': 'Wind speed x relative humidity / 100', 'Lat': 'Latitude', 'Lon': 'Longitude', 'LatPlusLon': 'Latitude + longitude', 'LatMinusLon': 'Latitude - longitude'}
def describe(n):
    if n in DESC: return DESC[n], ('Environmental' if n in ('Altitude', 'Temperature', 'Precipitation', 'Relative_humidity', 'Wind_speed') else 'Coordinates' if n.startswith('Lat') or n == 'Lon' else 'Climate index')
    m = re.match(r'kern(\d+)_(\d)', n)
    if m: return f'Gaussian-kernel {DIS[int(m.group(2))]} rate, bandwidth {m.group(1)} km', 'Kernel disease rate'
    m = re.match(r'knn(\d+)_(\d)', n)
    if m: return f'{DIS[int(m.group(2))]} rate over the {m.group(1)} nearest training locations', 'k-nearest-location rate'
    m = re.match(r'dens(\d+)', n)
    if m: return f'log kernel density of training households, bandwidth {m.group(1)} km', 'Sampling density'
    m = re.match(r'knndist(\d+)', n)
    if m: return f'log distance to the {m.group(1)}th nearest training location', 'Sampling density'
    return n, ''
tab('**Table S10.** Feature dictionary for FS5 (39 features; the first five form FS1)', ['No.', 'Feature', 'Group', 'Definition', 'Uses outcome labels'],
    [[i + 1, r.feature, describe(r.feature)[1], describe(r.feature)[0], 'Yes, training rows outside the own block' if r.uses_outcome_labels else 'No'] for i, r in FD.iterrows()],
    [500, 1500, 1800, 4000, 1900], size=14, note='Disease rates are computed for all three diseases, so each disease model receives its own rates and the rates of the other two diseases.')
DV = pd.read_csv(TAB / 'data_validation_report_v1.csv'); DV.columns = ['Check', 'Result']
tab('**Table S11.** Data validation checks', ['Check', 'Result'], DV.values.tolist(), [5400, 4200], size=16)

D4 = pd.read_csv(DC / 'D4_variogram_ranges_v1.csv')
tab('**Table S12.** Practical ranges of exponential variograms (location level, 20 km bins up to 600 km)', ['Variable', 'Uses outcome labels', 'Practical range (km)', 'Range beyond the 600 km fitting window'],
    [[r.variable, 'Yes' if r.uses_outcome_labels else 'No', f'{r.practical_range_km:.0f}', 'Yes' if r.beyond_600km_fit_window else 'No'] for _, r in D4.iterrows()], [4200, 1600, 1800, 2000],
    note='The 81 km block size is the median of the three outcome ranges (all years), computed in an earlier analysis with all labels. Ranges beyond the fitting window are not well determined.')
D5 = pd.read_csv(DC / 'D5_campaign_descriptors_v1.csv'); VS = pd.read_csv(DC / 'D5_variance_shares_v1.csv')
tab('**Table S13.** Survey campaigns: coverage, overlap with other campaigns and spread of location-level prevalence', ['Year', 'Households', 'Locations', 'Regions', 'Locations with another campaign within 81 km (%)', 'Median distance to nearest other-campaign location (km)'] + [f'{d} location prevalence IQR' for d in DIS],
    [[r['Survey year'], r.Households, r.Locations, r.Regions, r['Locations with another campaign within 81 km (%)'], r['Median distance to nearest other-campaign location (km)']] + [r[f'{d} location prevalence IQR (locations with >= 5 households)'] for d in DIS] for _, r in D5.iterrows()],
    [600, 850, 800, 1900, 1150, 1150, 1050, 1050, 1050], size=13,
    note='IQR: interquartile range of location prevalence among locations with at least five households. Share of household-level outcome variance explained (one-way eta squared): survey year ' + ', '.join(f"{r.disease} {100 * r.eta2_survey_year:.1f}%" for _, r in VS.iterrows()) + '; location within year ' + ', '.join(f"{r.disease} {100 * r.eta2_location_within_year:.1f}%" for _, r in VS.iterrows()) + '.')
OPT = pd.read_csv(TAB / 'TableS_optimisation_stage5_as_run_v1.csv')
tab('**Table S14.** Optimisation study (Section S2): swarm and evolutionary optimisation of feature and learner settings, as run before the code corrections (development repeat 1, 15 runs per optimiser and disease). Not used in the paper.',
    ['Disease', 'Optimiser', 'Inner fitness mean (SD)', 'Outer ROC-AUC mean (SD)', 'Outer PR-AUC', 'Distinct candidates', 'FS5 fixed', 'Rank average'],
    [[r.disease, r.optimiser, f'{r.inner_fitness_mean:.3f} ({r.inner_fitness_sd:.3f})', f'{r.outer_ROC_AUC_mean:.3f} ({r.outer_ROC_AUC_sd:.3f})', f3(r.outer_PR_AUC_mean), f'{r.unique_candidates:.0f}',
      f3(r['reference: FS5 fixed features, ExtraTrees-300']), f3(r['reference: (a) simple rank average'])] for _, r in OPT.iterrows()],
    [800, 1000, 1500, 1500, 1000, 1100, 1000, 1100], note='Each run used 10 agents and 150 fitness evaluations; fitness was mean minus 0.5 SD of ROC-AUC over 5 inner spatial folds. FS5 fixed and rank average are reference values on the same outer folds (pre-correction code).')


def f2(s): return s
AB = pd.read_csv(T2D / 'R2T1_ablation_performance_v1.csv')
tab('**Table S15.** Feature ablation: performance of ExtraTrees with the nested feature sets FS1 to FS5 (development, 15 splits)', list(AB.columns), AB.values.tolist(),
    [700, 900, 800, 1500, 1700, 1600, 1400], note='Pooled ROC-AUC: out-of-fold predictions of each repeat pooled, averaged over repeats; 95% CI from 1,000 location-cluster bootstrap resamples. Within-year AUC follows Eq. (2) of the paper.')
AI = pd.read_csv(T2D / 'R2T2_ablation_increments_v1.csv')
tab('**Table S16.** Feature ablation: change in ROC-AUC when each feature group is added (development, 15 splits)', list(AI.columns), AI.values.tolist(),
    [800, 1700, 650, 1350, 650, 550, 1350, 1400, 750], size=13, note='Fold mean: corrected resampled t 95% CI over the 15 folds. Pooled: location-cluster bootstrap. Holm correction across the 12 steps.')
SG = pd.read_csv(T2D / 'R2T4_shap_group_importance_v1.csv')
tab('**Table S17.** Feature-group importance of the FS5 ExtraTrees model: summed mean absolute SHAP value (development repeat 1) and grouped permutation importance (15 folds)',
    ['Disease', 'Feature group', 'Features', 'Summed mean |SHAP|', 'Permutation ROC-AUC drop'], [[r.disease, r.group, r.features, f'{r.mean_abs_SHAP_sum:.4f}', f'{r.permutation_AUC_drop:+.4f}'] for _, r in SG.iterrows()],
    [900, 2400, 900, 1800, 2000], note='SHAP values are on the probability scale. Spearman correlation between the two group rankings: ' + ', '.join(f'{k} {v:.2f}' for k, v in json.load(open(R2 / 'rev2_analysis_summary_v1.json'))['shap_vs_permutation_spearman'].items()) + ' (six groups). The values for every feature are in the results folder (R2T3_shap_feature_importance_v1.csv).')
AO = pd.read_csv(T2D / 'R2T5_area_of_applicability_v1.csv')
tab('**Table S18.** Area of applicability at surveyed test locations (ExtraTrees, development, 15 splits)', list(AO.columns), AO.values.tolist(),
    [700, 700, 1000, 1000, 800, 900, 900, 900, 1100, 1000], size=13, note='DI: dissimilarity index. ROC-AUC outside the AOA rests on 3-7 positive households and is not interpretable. Households outside: mean per repeat. Spearman correlation between DI and the absolute difference between the outcome and the model score (score expressed as a quantile of the training out-of-fold scores).')
CI6 = pd.read_csv(T2D / 'R2T6_metrics_with_CI_v1.csv')
tab('**Table S19.** Probability and threshold metrics with 95% location-cluster bootstrap intervals (pooled development predictions)', list(CI6.columns), CI6.values.tolist(),
    [600, 1200, 700, 1050, 1050, 1100, 1100, 1050, 1050, 1050, 1050, 1050], size=12, note='Pooled out-of-fold predictions of each repeat; estimates averaged over the three repeats; 1,000 location-cluster bootstrap resamples. Fold-level medians are in Table S1 and in the calibration table of the paper.')
CID = pd.read_csv(T2D / 'R2T7_campaign_identification_v1.csv')
tab('**Table S20.** Survey-year identification from the features (development, 15 splits)', list(CID.columns), CID.values.tolist(), [2000, 2000, 1000, 800, 800, 800, 800, 800],
    note='Difference in macro ROC-AUC, neighbourhood rates only minus FS1: ' + '{:+.3f} (corrected-t 95% CI {:+.3f} to {:+.3f}; p = {:.2f}).'.format(*json.load(open(R2 / 'rev2_analysis_summary_v1.json'))['rates_minus_FS1_year_AUC']))
NG = pd.read_csv(T2D / 'R2T8_ensemble_without_groupbalanced_v1.csv')
tab('**Table S21.** Rank-average ensemble with and without the group-balanced CatBoost model (development, 15 splits)', list(NG.columns), NG.values.tolist(), [800, 1400, 1600, 1800, 1200, 2000])
DS = pd.read_csv(T2D / 'R2T9_train_test_distances_v1.csv')
tab('**Table S22.** Distance from each test household to the nearest training location', list(DS.columns), DS.values.tolist(), [2600, 1300, 1000, 1300, 1300, 1400],
    note='Random CV: household-level 5-fold split; spatial: all 15 development splits (each household appears once per repeat). Buffers remove training locations within the given distance of any test location.')
SF = pd.read_csv(T2D / 'R2T3_shap_feature_importance_v1.csv')
rows = []
for d in DIS:
    q = SF[SF.disease == d].sort_values('mean_abs_SHAP', ascending=False).head(10)
    rows += [[d, i + 1, r.label, r.group, f'{r.mean_abs_SHAP:.4f}', f'{r.impurity_importance:.4f}'] for i, (_, r) in enumerate(q.iterrows())]
tab('**Table S23.** Ten features with the largest mean absolute SHAP value for the FS5 ExtraTrees model (development repeat 1)', ['Disease', 'Rank', 'Feature', 'Group', 'Mean |SHAP|', 'Impurity importance'], rows, [900, 600, 2800, 2200, 1400, 1600], size=13)


# ---------------------------------------------------------------- tables moved from the main text in the IJIES version
F2 = pd.read_csv(T2D / 'Table2_feature_sets_v2.csv')
tab('**Table S24.** Nested feature sets used in the ablation', list(F2.columns), F2.values.tolist(), [700, 1900, 4300, 650, 650, 2200], size=13)
M3 = pd.read_csv(T2D / 'Table3_models_v3.csv')
tab('**Table S25.** Models and fixed hyperparameters (read from the model objects)', list(M3.columns), M3.values.tolist(), [3000, 2000, 4700], size=13,
    note='Library feature sets: B = 5 environmental variables; E = environmental, climate and coordinate terms (14); EL = E plus kernel rates and densities (26); ELK = EL plus k-nearest rates and distance (36); G = location-level inputs for kriging. The number of library members is given in brackets (23 in total).')
CA = pd.read_csv(T2D / 'R2T11_calibration_fold_and_pooled_v1.csv')
tab('**Table S26.** Calibration within folds and on pooled predictions (development, 15 splits)', ['Disease', 'Model', 'Slope: fold median (IQR)', 'Slope: pooled [95% CI]', 'Intercept: pooled [95% CI]', 'Brier: pooled [95% CI]', 'Reference Brier p(1 - p)', 'Brier skill score [95% CI]'],
    [[r.Disease, r.Model.replace(' Rank average', '').replace(' Inner-selected single', ''), r['Calibration slope: fold median (IQR)'], r['Calibration slope: pooled [95% CI]'], r['Calibration intercept: pooled [95% CI]'], r['Brier: pooled [95% CI]'], r['Reference Brier p(1-p)'], r['Brier skill score [95% CI]']] for _, r in CA.iterrows()],
    [750, 650, 1550, 1450, 1500, 1450, 1000, 1550], size=12,
    note='Fold median: median over the 15 test folds. Pooled: out-of-fold probabilities of each repeat pooled; 95% CI from 1,000 location-cluster bootstrap resamples. Slopes above 1 mean the probabilities are too close to the average risk. Brier skill score = 1 - Brier / p(1 - p); its interval is derived from the Brier interval with the reference held fixed.')
P7 = pd.read_csv(TAB / 'Table7_disease_specific_comparisons_v1.csv')
tab('**Table S27.** Paired ROC-AUC differences between models (development, 15 splits)', ['Comparison', 'Disease', 'Overall: fold mean [corrected-t 95% CI]', 'Folds > 0', 'Holm p', 'Overall: pooled [bootstrap 95% CI]', 'Within-year: fold mean [corrected-t 95% CI]', 'Within-year: pooled [bootstrap 95% CI]'],
    [[q.Comparison.replace(' Rank average', ''), q.Disease, q['Overall: fold mean [corrected-t 95% CI]'], q['Overall: folds > 0'], f"{q['Overall: Holm p']:.3f}", q['Overall: pooled [bootstrap 95% CI]'], q['Within-year: fold mean [corrected-t 95% CI]'], q['Within-year: pooled [bootstrap 95% CI]']] for _, q in P7.iterrows()],
    [1400, 700, 1700, 750, 700, 1650, 1700, 1650], size=12, note='Holm correction across the three diseases within each comparison. Within-year differences use Eq. (2) of the paper. Fold-by-fold values: Figure S5.')
CS = pd.read_csv(REV / 'tables' / 'RT2_campaign_specific_vs_pooled_v1.csv')
tab('**Table S28.** Within-campaign ROC-AUC of models trained on all campaigns (pooled) and on one campaign at a time (development, 15 splits, ExtraTrees)', list(CS.columns), [[f'{v:.3f}' if isinstance(v, float) else v for v in row] for row in CS.values.tolist()],
    [800, 800, 1000, 1500, 1600, 2300, 1300], size=12, note='Fold-level ROC-AUC within each survey year, weighted by positive × negative pairs and averaged over the 15 folds; only year-fold test sets with at least five positive and five negative households. 95% CI: corrected resampled t.')

# ------------------------------------------------------------------ figures
add('h1', 'S4. Supplementary figures')
fig('FigS1_ROC_PR_curves_v1', '**Figure S1.** ROC and precision-recall curves for pooled out-of-fold predictions (development repeat 1).')
fig('FigS2_calibration_v2', '**Figure S2.** Calibration plots of the Platt-calibrated probabilities by deciles of predicted probability (pooled development predictions, repeat 1).')
fig('FigS3_confusion_matrices_v1', '**Figure S3.** Confusion matrices at the Youden threshold chosen on inner out-of-fold data (pooled development predictions).')
fig('FigS4_grouped_importance_v2', '**Figure S4.** Grouped permutation importance for the FS5 ExtraTrees model (mean drop in ROC-AUC over 15 development folds).')
fig('Fig4_FS1_vs_FS5_paired_folds_v2', '**Figure S5.** FS1 against FS5 ROC-AUC in each of the 15 development folds. Points above the dashed line favour FS5; intervals are corrected resampled t 95% CIs.', folder=RFG)
fig('Fig5_permutation_null_1000_v2', '**Figure S6.** Null distributions of the FS5 minus FS1 ROC-AUC under household-level and village-level label permutation (1,000 permutations per scheme); the orange line and the text in each panel give the observed value and its permutation p-value (development repeat 1).', folder=ROOT / 'results/paper1_rev5_v1/figures')
fig('R2Fig5_calibration_with_bands_v1', '**Figure S7.** Calibration of the FS5 and rank-average probabilities by deciles of predicted probability (development repeat 1), with 95% location-cluster bootstrap bands.', folder=F2D)
fig('R2Fig3_partial_dependence_v1', '**Figure S8.** Partial dependence of the FS5 ExtraTrees model on the three features with the largest mean absolute SHAP value for each disease, computed on the training data of each fold of development repeat 1 (one line per fold).', folder=F2D)
fig('R2Fig6_train_test_distances_v2', '**Figure S9.** Cumulative distribution of the distance from each test household to the nearest training location under random cross-validation, spatial blocks of 81, 120 and 160 km, and 81 km blocks with 50 km and 100 km buffers.', folder=F2D)
fig('Fig3_random_vs_spatial_cv_v1', '**Figure S10.** Mean ROC-AUC under random 5-fold cross-validation and spatial block cross-validation (81 km blocks), with the same corrected code. The dotted line marks chance level.', folder=RFG)
fig('R2Fig4_area_of_applicability_FS5_v1', '**Figure S11.** Dissimilarity index of the FS5 ExtraTrees model divided by the area-of-applicability threshold, at surveyed locations (mean over development repeats). Orange rings: locations outside the area of applicability in at least one repeat. Values describe how unusual the predictors are, not disease risk.', folder=F2D)
fig('Fig4_spatial_sensitivity_v2', '**Figure S12.** FS5 minus FS1 ROC-AUC under proximity removal (blue), size-matched random removal (grey) and larger blocks (green). Bars: mean of 15 development folds; whiskers: corrected resampled t 95% intervals.')
fig('Fig3_overall_vs_within_year_AUC_v1', '**Figure S13.** Overall (filled) and within-survey-year (open) pooled ROC-AUC with 95% location-cluster bootstrap intervals. (a) rank average; (b) inner-selected single model. The dotted line marks chance level.')
fig('Fig6_campaign_specific_vs_pooled_v3', '**Figure S14.** Difference in within-campaign ROC-AUC between campaign-specific and pooled models, with corrected resampled t 95% intervals; numbers on the right give the pooled and campaign-specific values.', width=520, folder=RFG)
fig('Fig8_forward_temporal_v1', '**Figure S15.** (a) Prevalence in the 2012-2015 training campaigns and the 2018-2019 test campaigns. (b) Forward temporal ROC-AUC with 95% location-cluster bootstrap intervals; the dotted line marks chance level.', folder=RFG)

# references cited in the supplement
for b in B:
    if 'text' in b:
        def rep(m):
            k = m.group(1)
            if k not in order: order.append(k)
            return f'[S{order.index(k) + 1}]'
        b['text'] = re.sub(r'(?<![\^_])\{([A-Za-z][A-Za-z0-9]*)\}', rep, b['text'])
    if 'text' in b: b['text'] = re.sub(r'(\d) x (\d)', '\\1 × \\2', b['text'])
if order:
    add('h1', 'Supplementary references')
    for i, k in enumerate(order, 1): add('reference', f'[S{i}] ' + REFS.loc[k, 'reference_IEEE'])
spec = {'title': 'Supplementary Material', 'running_head': 'Supplementary Material: spatial generalisation and survey-campaign confounding', 'blocks': B}
json.dump(spec, open(ROOT / 'manuscript' / 'supplement_ijies_spec_v2.json', 'w'), indent=1, ensure_ascii=False); print('supplement blocks', len(B))
