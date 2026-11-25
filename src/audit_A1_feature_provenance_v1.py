"""Feature-leakage audit, stage A1 (2026-11-21): feature provenance table.

Builds one row per feature that any model in the study can receive, using the feature NAMES produced by the actual code
(new pipeline: common.features / common.fitness; old pipeline: data/reference/run3_cattle_v5_pipeline_v3.py, which
feeds the 23-model rank average). Definitions, source rows/labels, restrictions and priors are traced from the code
(line references in the 'code' column). Verification status here reflects only what A1 itself executes:
  * label-free features: an executed check that their values do not change when all labels are permuted (Passed/Failed);
  * label-derived features: 'Unresolved' until A2 (independent re-implementation), A3 (label perturbation) and A4
    (negative controls) have been run. Temporal status is recorded from the data and is examined in A7.
Development data only; nothing is fitted. Run from the project root: python src/audit_A1_feature_provenance_v1.py"""
import sys, json, shutil, warnings
from pathlib import Path
import numpy as np, pandas as pd
SRC = Path(__file__).resolve().parent; sys.path.insert(0, str(SRC)); ROOT = SRC.parent
from common.data import load, labels, ENV, EXCLUDED, DIS
from common.features import climate_coords, CLIMATE_NAMES, COORD_NAMES, Neighbourhood
from common.fitness import FeatureBuilder
warnings.filterwarnings('ignore')
OUT = ROOT / 'results/audit_v1/A1'; OUT.mkdir(parents=True, exist_ok=False)

df = load(ROOT / 'data/raw'); Y = labels(df); FB = FeatureBuilder(df, Y, Neighbourhood(df, Y), climate_coords(df))

# ---- actual feature names from the NEW pipeline (FS5 + IDW, which exists only in the optimisers' search space) ----
FS5 = {'climate': CLIMATE_NAMES, 'coordinates': True, 'kernel_rates': True, 'knn_rates': True, 'idw_rates': True, 'density_terms': True,
       'cross_disease_rates': True, 'bandwidths': (50.0, 150.0, 400.0), 'ks': (5, 15, 40), 'k': 15, 'prior': 1.0, 'idw_power': 2.0,
       'min_samples_leaf': 10, 'max_features': 'sqrt', 'complexity': 1.0}
tr = np.where(df.spatial_fold_r1.values != 0)[0]; te = np.where(df.spatial_fold_r1.values == 0)[0]
_, _, NEW_NAMES = FB.build(FS5, tr, te, df.block_r1.values, 1)          # disease index 1 = FMD; cross-disease on -> all 3 rates

# ---- actual feature names from the OLD pipeline (run-3 code, executed up to its main loop, as Stage 9 does) ----
src = open(ROOT / 'data/reference/run3_cattle_v5_pipeline_v3.py').read()
src = src[:src.index("t0 = time.time()\nRES = sorted(")].replace("DATA = Path('data')", "DATA = Path('data/raw')").replace("OUT = next_free_dir('results')", "OUT = next_free_dir('_audit_scratch')")
import os; cwd = os.getcwd(); os.chdir(ROOT)
R3 = {'__name__': 'run3'}; exec(compile(src, 'run3_cattle_v5_pipeline_v3.py', 'exec'), R3); shutil.rmtree(ROOT / R3['OUT'], ignore_errors=True); os.chdir(cwd)
assert (R3['df'].ID.values == df.ID.values).all() and (R3['Y_ALL'] == Y).all()
OLD_NAMES = list(R3['NAMES_E']) + list(R3['NAMES_LAG']) + list(R3['NAMES_KNN'])          # feature set ELK (36)

# ---- executed check for label-free features: values must not change when all three labels are permuted ----
rng = np.random.default_rng(20261121); Yp = Y[rng.permutation(len(Y))]
cc = climate_coords(df); cc_perm = climate_coords(df.assign(**{d: Yp[:, k] for k, d in enumerate(DIS)}))
label_free_check = {c: bool(np.array_equal(cc[c].values, cc_perm[c].values)) for c in cc.columns}
F1, _, names1 = FB.build({**FS5, 'kernel_rates': True, 'knn_rates': True}, tr, te, df.block_r1.values, 1)
FBp = FeatureBuilder(df, Yp, Neighbourhood(df, Yp), cc); F2, _, names2 = FBp.build({**FS5}, tr, te, df.block_r1.values, 1)
for j, nm in enumerate(names1):
    if nm.startswith('dens') or nm.startswith('knndist'): label_free_check[nm] = bool(np.array_equal(F1[:, j], F2[:, j]))

years = df.Year.value_counts().sort_index().to_dict()
TEMPORAL_POOLED = f'None: all survey years pooled ({", ".join(f"{y}: {n}" for y, n in years.items())} households); 13 of 462 locations surveyed in >1 year. Examined in A7.'
TEMPORAL_NA = 'Not applicable (no labels used); environmental values as supplied in the dataset (their reference period must be taken from the ILRI metadata).'
EXCL_NEW = 'All fit (training) locations in the QUERY location\'s own spatial block are excluded (81 km blocks of the active partition); for a test location its own block is a test block, so no fit row is removed by this rule.'
PRIOR_NEW = 'p0 = positive rate of fit rows OUTSIDE the query location\'s own block (features.py L48-L50); pseudo-count = prior (1.0 in FS5).'
PRIOR_OLD = 'p0 = positive rate of ALL fit rows (run3 L167/L185). For a TRAINING query this includes its own block and its own household (weight about 1/2,670); for a TEST query p0 uses training rows only. Known issue, examined in A5.'
SRC_ROWS = 'Fit (training) households of the current outer or inner split only: n and pos are built from Y[fit_idx] (features.py L46-L47; run3 L165-L166/L183-L184).'
AVAIL = 'Available at prediction time under Scenario A (retrospective spatial prediction from pooled survey data of other areas). Requires surveyed locations outside the query block.'

rows = []
def add(name, cat, definition, src_obs, src_lab, geo, temporal, prior, avail, risk, status, code, pipeline):
    rows.append({'Feature name': name, 'Pipeline': pipeline, 'Feature category': cat, 'Mathematical definition': definition, 'Source observations': src_obs,
                 'Source labels': src_lab, 'Geographic restriction': geo, 'Temporal restriction': temporal, 'Prior information': prior,
                 'Training/test availability': avail, 'Leakage risk (before tests)': risk, 'Verification result (A1)': status, 'Code': code})

LF = lambda nm: 'Passed: unchanged under full label permutation (executed)' if label_free_check.get(nm) else 'FAILED: changed under label permutation'
DEFS = {'SL_temperature': 'Temperature + 0.0065 x Altitude', 'DeMartonne': 'Precipitation / (Temperature + 10)', 'Temp_x_RH': 'Temperature x RH / 100',
        'Precip_x_RH': 'log(1 + Precipitation) x RH / 100', 'log_Precip': 'log(1 + Precipitation)', 'Wind_x_RH': 'Wind_speed x RH / 100',
        'Lat': 'Latitude', 'Lon': 'Longitude', 'LatPlusLon': 'Latitude + Longitude', 'LatMinusLon': 'Latitude - Longitude'}
for nm in ENV:
    add(nm, 'Original environmental', 'As supplied in the dataset (household row)', 'The household itself', 'None', 'None', TEMPORAL_NA, 'None',
        'Available for any location with the environmental layers', 'Low', LF(nm), 'data.py L22-L40; features.py L11-L22', 'new + old (B, E, EL, ELK)')
for nm in CLIMATE_NAMES[5:]:
    add(nm, 'Climate index (fixed formula)', DEFS[nm], 'The household itself', 'None', 'None', TEMPORAL_NA, 'None', 'Available wherever the 5 variables are',
        'Low', LF(nm), 'features.py L11-L22; run3 L140-L146' + (' (new pipeline only)' if nm == 'Precip_x_RH' else ''), 'new + old' if nm != 'Precip_x_RH' else 'new')
for nm in COORD_NAMES:
    add(nm, 'Coordinate term', DEFS[nm], 'The household itself', 'None', 'None', 'Not applicable', 'None', 'Available for any location', 'Low', LF(nm),
        'features.py L21; run3 L146', 'new + old')

for nm in NEW_NAMES:
    if nm in CLIMATE_NAMES or nm in COORD_NAMES: continue
    if nm.startswith('kern') or nm.startswith('knn') and not nm.startswith('knndist') or nm.startswith('idw'):
        d = int(nm.split('_')[-1]); dn = DIS[d]
        cat = 'Neighbourhood disease rate' if d == 1 else 'Cross-disease rate (for FMD models; own-disease rate for the ' + dn + ' model)'
        if nm.startswith('kern'):
            b = nm[4:].split('_')[0]
            dfn = f'r(x) = [sum_j K(x,j) pos_{dn}(j) + c p0(x)] / [sum_j K(x,j) n(j) + c], K = exp(-dist^2 / (2 x {b}^2)), j = fit locations outside block(x); c = prior'
        elif nm.startswith('knn'):
            k = nm[3:].split('_')[0]
            dfn = f'r(x) = [sum over the {k} nearest fit locations outside block(x) of pos_{dn} + c p0(x)] / [their households + c]; p0(x) if none available'
        else:
            dfn = f'r(x) = [sum_j w_j pos_{dn}(j) + c p0(x)] / [sum_j w_j n(j) + c], w_j = 1 / max(dist,1)^p (search space only, not in FS5)'
        add(nm, cat, dfn, SRC_ROWS, f'{dn} labels of fit households', EXCL_NEW, TEMPORAL_POOLED, PRIOR_NEW, AVAIL,
            'Medium (label-derived; spatial dependence; temporal pooling)', 'Unresolved: pending A2, A3, A4 (A6, A7 for scope)', 'features.py L41-L72', 'new (FS5' + (')' if not nm.startswith('idw') else ' no; search space only)'))
    elif nm.startswith('dens') or nm.startswith('knndist'):
        dfn = 'log(1 + sum_j K(x,j) n(j)) over fit locations outside block(x)' if nm.startswith('dens') else 'log(1 + distance to the k-th nearest fit location outside block(x))'
        add(nm, 'Sampling-density term', dfn, 'Counts of fit households (no labels)', 'None', EXCL_NEW, 'All survey years pooled (counts only)', 'None',
            AVAIL, 'Low (label-free; reflects survey design)', LF(nm), 'features.py L57, L66', 'new')

for nm in R3['NAMES_LAG'] + R3['NAMES_KNN']:
    if 'density' in nm or 'log_dist' in nm:
        add(nm, 'Sampling-density term', 'log(1 + kernel-weighted fit households outside block(x))' if 'density' in nm else 'log(1 + distance to the 15th fit location outside block(x))',
            'Counts of fit households (no labels)', 'None', EXCL_NEW, 'All survey years pooled (counts only)', 'None', AVAIL, 'Low (label-free)',
            'Unresolved: label-free by code trace; executed check in A3', 'run3 L162-L198', 'old (EL/ELK, 23-model library)')
        continue
    dn = [x for x in DIS if x in nm][0]
    add(nm, 'Neighbourhood / cross-disease rate (old code)', f'As the new kernel/kNN rate for {dn}, prior strength 1', SRC_ROWS, f'{dn} labels of fit households', EXCL_NEW,
        TEMPORAL_POOLED, PRIOR_OLD, AVAIL, 'Medium (label-derived; known own-block prior issue for training rows)', 'Unresolved: pending A2, A3, A4, A5',
        'run3 L162-L198', 'old (EL/ELK, 23-model library)')

for nm in EXCLUDED + ['Weight_CBPP', 'Weight_FMD', 'Weight_LSD', 'ID', 'Location_id', 'fold / block columns']:
    used = nm in NEW_NAMES or nm in OLD_NAMES
    add(nm, 'Excluded variable', 'Never a predictor' + (' (outcome-derived: Weight)' if 'Weight' in nm else ''), '-', '-', '-', '-', '-', '-',
        'High if used' if 'Weight' in nm else 'Design variable', 'FAILED: present in a feature list' if used else 'Passed: absent from every feature list (executed)',
        'data.py L9', 'none')

T = pd.DataFrame(rows); T.to_csv(OUT / 'A1_feature_provenance_v1.csv', index=False)
other_uses = pd.DataFrame([
    {'Item': 'GP-kriging [E] (old library model)', 'Uses labels': 'Yes: empirical logit of fit locations (training only)', 'Note': 'Model, not a feature; scaler fitted on fit locations (run3 L243-L259). Checked in A10.'},
    {'Item': 'CatBoost-groupbalanced [EL]', 'Uses labels': 'Training labels', 'Note': 'Year_region used ONLY as training sample weights (run3 L261, L274), never as a feature. Checked in A10.'},
    {'Item': 'CatBoost-village [EL]/[ELK]', 'Uses labels': 'Training village rates', 'Note': 'Trained on fit-location means; predicts household rows. Checked in A10.'},
    {'Item': 'ECDF rank scaling of the 23 models', 'Uses labels': 'No (training OOF scores only)', 'Note': 'run3 L407-L412; Stage 9 ecdf_rank_average. Checked in A10.'},
    {'Item': 'Threshold rule', 'Uses labels': 'Training labels', 'Note': 'Youden index on inner out-of-fold predictions (experiment_config.yaml L57). Checked in A10.'}])
other_uses.to_csv(OUT / 'A1_label_uses_outside_features_v1.csv', index=False)
summ = {'new_pipeline_FS5_plus_idw_features': len(NEW_NAMES), 'old_pipeline_ELK_features': len(OLD_NAMES), 'rows_in_table': len(T),
        'label_free_checks_passed': int(sum(label_free_check.values())), 'label_free_checks_total': len(label_free_check),
        'excluded_variables_found_in_feature_lists': [n for n in EXCLUDED if n in NEW_NAMES or n in OLD_NAMES],
        'survey_years_households': years, 'risk_counts': T['Leakage risk (before tests)'].str.split(' ').str[0].value_counts().to_dict(),
        'status_counts': T['Verification result (A1)'].str.split(':').str[0].value_counts().to_dict()}
json.dump(summ, open(OUT / 'A1_summary_v1.json', 'w'), indent=1); print(json.dumps(summ, indent=1))
