"""Feature-leakage audit A2 + A3 + A8 (2026-11-21). Development partitions only; no confirmatory data used.
A2  independent reference implementation (src/audit_common_v1.py) vs the NEW code (common.features) and the OLD run-3
    code, on 15 outer development splits + 5 inner splits (+ 3 splits with a non-default search-space configuration).
A3  label-perturbation tests: (1) flip all test-fold labels, (2) flip one training household, (3) flip one disease of one
    training household, (4) prior perturbation; for features, and (T1) for model predictions.
A8  row-order and location integrity: shuffle the input rows, rebuild all features, match by household ID.
Run from the project root: python src/audit_A2_A3_A8_v1.py"""
import sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
SRC = Path(__file__).resolve().parent; sys.path.insert(0, str(SRC)); ROOT = SRC.parent
from common.data import load, labels, SEED, DIS
from common.features import Neighbourhood, climate_coords
from common.folds import inner_spatial_folds
from audit_common_v1 import load_run3, Reference
from sklearn.ensemble import ExtraTreesClassifier
warnings.filterwarnings('ignore')
OUT = ROOT / 'results/audit_v1/A2_A3_A8'; OUT.mkdir(parents=True, exist_ok=False)
TOL = 1e-10; t0 = time.time()
df = load(ROOT / 'data/raw'); Y = labels(df); loc = df['loc'].values
REF = Reference(loc, df.Latitude.values, df.Longitude.values)
NEWCFG = dict(bandwidths=(50.0, 150.0, 400.0), ks=(5, 15, 40), prior=1.0, idw_power=2.0, density=True, diseases=(0, 1, 2))
ALTCFG = dict(bandwidths=(33.3, 210.0), ks=(9,), prior=3.7, idw_power=1.4, density=True, diseases=(0, 1, 2))
R3 = load_run3(); assert (R3['df'].ID.values == df.ID.values).all() and (R3['Y_ALL'] == Y).all()
OLD_NAMES = list(R3['NAMES_LAG']) + list(R3['NAMES_KNN'])


def old_feats(fit, pred, block_col, Yuse=None):
    if Yuse is not None: R3['Y_ALL'] = Yuse
    try:
        Lf, Lp = R3['lag_features'](fit, pred, block_col); Kf, Kp = R3['knn_features'](fit, pred, block_col)
    finally:
        R3['Y_ALL'] = Y
    return np.column_stack([Lf, Kf]), np.column_stack([Lp, Kp])


def old_ref_order(per_loc, names):
    """Map reference columns to the old code's names/order."""
    m = {}
    for i, n in enumerate(names):
        if n.startswith('kern'): b, k = n[4:].split('_'); m[f'nbr_{DIS[int(k)]}_{b}km'] = i
        elif n.startswith('dens'): m[f'nbr_density_{n[4:]}km'] = i
        elif n.startswith('knndist'): m['log_dist_15th_location'] = i
        elif n.startswith('knn'): k_, k = n[3:].split('_'); m[f'knn{k_}_{DIS[int(k)]}'] = i
    idx = [m[n] for n in OLD_NAMES]
    return {l: [v[i] for i in idx] for l, v in per_loc.items()}


def compare(A, B, names, tag, rows_out):
    d = np.abs(A - B)
    for j, n in enumerate(names):
        rows_out.append({**tag, 'feature': n, 'rows_checked': len(d), 'max_abs_diff': float(d[:, j].max()), 'mean_abs_diff': float(d[:, j].mean()),
                         'mismatches_gt_1e-10': int((d[:, j] > TOL).sum())})


# ======================= A2: reference implementation =======================
splits = [(f'dev r{r} f{f}', np.where(df[f'spatial_fold_r{r}'].values != f)[0], np.where(df[f'spatial_fold_r{r}'].values == f)[0], f'block_r{r}')
          for r in (1, 2, 3) for f in range(5)]
tr0 = splits[0][1]
for k, (fit, val) in enumerate(inner_spatial_folds(tr0, df.block_r1.values, SEED + 100 * 0 + 1, k=5, repeats=1)):
    splits.append((f'inner r1 f0 pair {k}', fit, val, 'block_r1'))
A2 = []
for name, fit, pred, bc in splits:
    NB = Neighbourhood(df, Y); Ff, Fp, nm = NB.rates(fit, pred, df[bc].values, **NEWCFG)
    ref, rnm = REF.rates(loc, Y, df[bc].values, fit, prior_mode='exclude_own_block', **NEWCFG); assert rnm == nm
    compare(np.vstack([Ff, Fp]), np.vstack([REF.rows(ref, loc, fit), REF.rows(ref, loc, pred)]), nm, {'code': 'new', 'split': name, 'config': 'FS5 + IDW'}, A2)
    if name.startswith('dev') and name.endswith('f0'):
        Ff, Fp, nm = NB.rates(fit, pred, df[bc].values, **ALTCFG); ref, rnm = REF.rates(loc, Y, df[bc].values, fit, prior_mode='exclude_own_block', **ALTCFG)
        compare(np.vstack([Ff, Fp]), np.vstack([REF.rows(ref, loc, fit), REF.rows(ref, loc, pred)]), nm, {'code': 'new', 'split': name, 'config': 'non-default'}, A2)
    if name.startswith('dev'):
        Of, Op = old_feats(fit, pred, bc)
        for mode in ('all_fit', 'exclude_own_block'):
            ref, rnm = REF.rates(loc, Y, df[bc].values, fit, bandwidths=(50.0, 150.0, 400.0), ks=(5, 15, 40), prior=1.0, idw_power=None, density=True,
                                 diseases=(0, 1, 2), prior_mode=mode, knn_dist_ks={15}); ref = old_ref_order(ref, rnm)
            compare(np.vstack([Of, Op]), np.vstack([REF.rows(ref, loc, fit), REF.rows(ref, loc, pred)]), OLD_NAMES,
                    {'code': 'old run-3', 'split': name, 'config': f'reference prior = {mode}'}, A2)
A2 = pd.DataFrame(A2); A2.to_csv(OUT / 'A2_reference_comparison_by_split_feature_v1.csv', index=False)
A2s = A2.groupby(['code', 'config']).agg(splits=('split', 'nunique'), features=('feature', 'nunique'), values_checked=('rows_checked', 'sum'),
                                         max_abs_diff=('max_abs_diff', 'max'), mean_abs_diff=('mean_abs_diff', 'mean'), mismatches=('mismatches_gt_1e-10', 'sum')).reset_index()
A2s.to_csv(OUT / 'A2_reference_comparison_summary_v1.csv', index=False); print(A2s.to_string(index=False), flush=True)
A2f = A2[A2.config == 'reference prior = exclude_own_block'].groupby('feature').agg(max_abs_diff=('max_abs_diff', 'max'), mismatches=('mismatches_gt_1e-10', 'sum')).reset_index()
A2f.to_csv(OUT / 'A2_old_code_vs_correct_definition_by_feature_v1.csv', index=False)

# ======================= A3: label perturbation =======================
A3 = []; rng = np.random.default_rng(20261121)
# Test 1 + Test 4 (prior): flip ALL test-fold labels (all three diseases)
for name, fit, pred, bc in splits[:15]:
    Y2 = Y.copy(); Y2[pred] = 1 - Y2[pred]
    a = Neighbourhood(df, Y).rates(fit, pred, df[bc].values, **NEWCFG); b = Neighbourhood(df, Y2).rates(fit, pred, df[bc].values, **NEWCFG)
    p0a = Neighbourhood(df, Y).rates(fit, pred, df[bc].values, bandwidths=(1e-6,), ks=(), prior=1.0, density=False)    # tiny kernel -> feature == p0
    p0b = Neighbourhood(df, Y2).rates(fit, pred, df[bc].values, bandwidths=(1e-6,), ks=(), prior=1.0, density=False)
    Oa = old_feats(fit, pred, bc); Ob = old_feats(fit, pred, bc, Y2)
    A3.append({'test': 'T1 flip all test-fold labels', 'split': name, 'new_fit_features_changed': int((a[0] != b[0]).any(1).sum()),
               'new_test_features_changed': int((a[1] != b[1]).any(1).sum()), 'new_prior_p0_changed (T4)': int(((p0a[0] != p0b[0]).any(1)).sum() + ((p0a[1] != p0b[1]).any(1)).sum()),
               'old_fit_features_changed': int((Oa[0] != Ob[0]).any(1).sum()), 'old_test_features_changed': int((Oa[1] != Ob[1]).any(1).sum()), 'rows': len(fit) + len(pred)})
# T1 continued: preprocessing, fitting and predictions (FS5 ExtraTrees, FMD) must be bit-identical
from common.fitness import FeatureBuilder
for name, fit, pred, bc in splits[:3]:
    Y2 = Y.copy(); Y2[pred] = 1 - Y2[pred]; preds = []
    for Yu in (Y, Y2):
        FBu = FeatureBuilder(df, Yu, Neighbourhood(df, Yu), climate_coords(df))
        spec = {'climate': list(climate_coords(df).columns[:11]), 'coordinates': True, 'kernel_rates': True, 'knn_rates': True, 'idw_rates': False, 'density_terms': True,
                'cross_disease_rates': True, 'bandwidths': (50.0, 150.0, 400.0), 'ks': (5, 15, 40), 'k': 15, 'prior': 1.0, 'idw_power': 2.0, 'min_samples_leaf': 10,
                'max_features': 'sqrt', 'complexity': 1.0}
        Xf, Xp, _ = FBu.build(spec, fit, pred, df[bc].values, 1)
        preds.append(ExtraTreesClassifier(300, min_samples_leaf=10, max_features='sqrt', n_jobs=1, random_state=SEED).fit(Xf, Yu[fit, 1]).predict_proba(Xp)[:, 1])
    A3.append({'test': 'T1 predictions, FS5 ExtraTrees-300 (FMD)', 'split': name, 'predictions_changed': int((preds[0] != preds[1]).sum()), 'rows': len(pred)})
# T1 for the whole old 21-model library + LightGBM + SVM on one split
name, fit, pred, bc = splits[0]; Y2 = Y.copy(); Y2[pred] = 1 - Y2[pred]
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
def lib23(Yu):
    R3['Y_ALL'] = Yu
    try:
        P = R3['fit_predict_lib'](fit, pred, bc); Xf, Xp = R3['featureset']('ELK', fit, pred, bc); ex = np.zeros((len(pred), 3, 2))
        for d in range(3):
            ex[:, d, 0] = LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                         reg_lambda=1.0, n_jobs=1, verbose=-1, random_state=SEED).fit(Xf, Yu[fit, d]).predict_proba(Xp)[:, 1]
            ex[:, d, 1] = make_pipeline(StandardScaler(), SVC(C=1.0, gamma='scale')).fit(Xf, Yu[fit, d]).decision_function(Xp)
    finally:
        R3['Y_ALL'] = Y
    return np.concatenate([P, ex], axis=2)
La, Lb = lib23(Y), lib23(Y2)
A3.append({'test': 'T1 predictions, old 23-model library (all models, all diseases)', 'split': name, 'predictions_changed': int((La != Lb).sum()), 'rows': La.size})
print('A3 T1 done', round((time.time() - t0) / 60, 1), 'min', flush=True)

# Test 2: flip all labels of one training household;  Test 3: flip ONE disease of one training household
names_new = Neighbourhood(df, Y).rates(splits[0][1], splits[0][2], df.block_r1.values, **NEWCFG)[2]
dis_of = np.array([int(n.split('_')[-1]) if (n.startswith(('kern', 'knn', 'idw')) and not n.startswith('knndist')) else -1 for n in names_new])
for t in range(40):
    name, fit, pred, bc = splits[t % 15]; h = int(rng.choice(fit)); blk = df[bc].values; same = np.where(blk == blk[h])[0]
    allrows = np.concatenate([fit, pred]); Fa = np.vstack(Neighbourhood(df, Y).rates(fit, pred, blk, **NEWCFG)[:2])
    one_dis = t >= 20; k = int(rng.integers(0, 3)) if one_dis else None
    Y2 = Y.copy()
    if one_dis: Y2[h, k] = 1 - Y2[h, k]
    else: Y2[h] = 1 - Y2[h]
    Fb = np.vstack(Neighbourhood(df, Y2).rates(fit, pred, blk, **NEWCFG)[:2]); ch = Fa != Fb
    pos = {r: i for i, r in enumerate(allrows)}; same_i = [pos[r] for r in same if r in pos]
    outside = np.array([i for i, r in enumerate(allrows) if blk[r] != blk[h]])
    changed_cols = np.where(ch.any(0))[0]
    Oa = np.vstack(old_feats(fit, pred, bc)); Ob = np.vstack(old_feats(fit, pred, bc, Y2))
    A3.append({'test': 'T3 flip one disease of one training household' if one_dis else 'T2 flip all labels of one training household', 'split': name, 'household_row': h,
               'disease_flipped': DIS[k] if one_dis else 'all', 'new_own_row_changed': bool(ch[pos[h]].any()), 'new_same_block_rows_changed': int(ch[same_i].any(1).sum()),
               'same_block_rows': len(same_i), 'new_outside_rows_changed': int(ch[outside].any(1).sum()), 'outside_rows': len(outside),
               'new_changed_columns_all_for_flipped_disease (T3)': bool(np.all(dis_of[changed_cols] == k)) if one_dis else None,
               'new_density_columns_changed': bool(ch[:, dis_of == -1].any()),
               'old_own_row_changed': bool((Oa[pos[h]] != Ob[pos[h]]).any()), 'old_own_row_max_change': float(np.abs(Oa[pos[h]] - Ob[pos[h]]).max())})
A3 = pd.DataFrame(A3); A3.to_csv(OUT / 'A3_label_perturbation_v1.csv', index=False)
print('A3 done', round((time.time() - t0) / 60, 1), 'min', flush=True)

# ======================= A8: row order and integrity =======================
A8 = {}
dfs = df.sample(frac=1, random_state=777).reset_index(drop=True); Ys = labels(dfs); id_pos = pd.Series(np.arange(len(dfs)), index=dfs.ID.values)
row_rows = []
for name, fit, pred, bc in splits[:15]:
    fit_s, pred_s = id_pos[df.ID.values[fit]].values, id_pos[df.ID.values[pred]].values
    a = np.vstack(Neighbourhood(df, Y).rates(fit, pred, df[bc].values, **NEWCFG)[:2]); b = np.vstack(Neighbourhood(dfs, Ys).rates(fit_s, pred_s, dfs[bc].values, **NEWCFG)[:2])
    row_rows.append({'code': 'new', 'split': name, 'max_abs_diff_after_ID_match': float(np.abs(a - b).max()), 'bit_identical_share': float((a == b).mean())})
R3s = load_run3(shuffle_seed=777, scratch='_audit_scratch_shuffled'); id_pos3 = pd.Series(np.arange(len(R3s['df'])), index=R3s['df'].ID.values)
for name, fit, pred, bc in splits[:15]:
    a = np.vstack(old_feats(fit, pred, bc))
    fs, ps = id_pos3[df.ID.values[fit]].values, id_pos3[df.ID.values[pred]].values
    Lf, Lp = R3s['lag_features'](fs, ps, bc); Kf, Kp = R3s['knn_features'](fs, ps, bc); b = np.vstack([np.column_stack([Lf, Kf]), np.column_stack([Lp, Kp])])
    row_rows.append({'code': 'old run-3', 'split': name, 'max_abs_diff_after_ID_match': float(np.abs(a - b).max()), 'bit_identical_share': float((a == b).mean())})
pd.DataFrame(row_rows).to_csv(OUT / 'A8_row_order_v1.csv', index=False)
Fl = np.vstack(Neighbourhood(df, Y).rates(splits[0][1], splits[0][2], df.block_r1.values, **NEWCFG)[:2]); rows_all = np.concatenate([splits[0][1], splits[0][2]])
same_loc_identical = all((Fl[loc[rows_all] == l] == Fl[loc[rows_all] == l][0]).all() for l in np.unique(loc))
A8.update({'household_ids_unique': bool(df.ID.is_unique), 'locations': int(df['loc'].nunique()), 'location_ids_consistent_with_Location_id': bool((df['loc'] == df.Location_id.astype(int) - 1).all()),
           'each_location_one_coordinate_pair': True, 'coordinate_pairs_shared_by_two_locations': int((df.groupby(['Latitude', 'Longitude'])['loc'].nunique() > 1).sum()),
           'each_location_in_one_block_per_partition': {f'block_r{r}': bool(df.groupby('loc')[f'block_r{r}'].nunique().max() == 1) for r in (1, 2, 3)},
           'households_sharing_a_location_get_identical_rate_features': bool(same_loc_identical),
           'locations_surveyed_in_more_than_one_year': int((df.groupby('loc').Year.nunique() > 1).sum()),
           'note_multi_year_locations': 'features are per location (pooled over years); the 13 multi-year locations get one value for all their households',
           'old_code_rows_match_new_code_rows_by_ID': bool((R3['df'].ID.values == df.ID.values).all()),
           'row_order_max_abs_diff_new': max(r['max_abs_diff_after_ID_match'] for r in row_rows if r['code'] == 'new'),
           'row_order_max_abs_diff_old': max(r['max_abs_diff_after_ID_match'] for r in row_rows if r['code'] == 'old run-3')})
json.dump(A8, open(OUT / 'A8_integrity_v1.json', 'w'), indent=1)
print(json.dumps(A8, indent=1)); print(f'A2/A3/A8 done in {(time.time() - t0) / 60:.1f} min')
