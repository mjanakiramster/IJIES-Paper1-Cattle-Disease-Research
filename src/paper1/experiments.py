"""Paper 1 experiments with the corrected code (deterministic kNN ties, corrected library prior). Every function writes a
checkpoint under results/paper1_v1/ and returns immediately if it already exists, so the notebook and the runner share
exactly the same code and an interrupted run can resume. Development folds only."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from .core import (setup, dev_splits, extratrees, load_library, FEATURE_SETS, OUT, DIS, SEED, auc, pr_auc)

E2 = OUT / 'E2_featuresets'; E3 = OUT / 'E3_library23'; E5 = OUT / 'E5_permutations'; E6 = OUT / 'E6_proximity_blocks'
E7 = OUT / 'E7_temporal'; E8 = OUT / 'E8_importance'
CACHE23 = Path(__file__).resolve().parents[2] / 'results/stage4_v1/cache/library23_predictions_v1.npz'
A5_PARTS = Path(__file__).resolve().parents[2] / 'results/audit_v1/A5/parts'
_S = {}


def _env():
    if 'df' not in _S:
        df, Y, NB, FB = setup(); _S.update(df=df, Y=Y, NB=NB, FB=FB)
    return _S['df'], _S['Y'], _S['NB'], _S['FB']


# ------------------------------------------------------------------ E2: FS1 vs FS5 (ExtraTrees-300), outer test + inner OOF
def e2_split(r, f):
    out = E2 / f'r{r}_f{f}.npz'
    if out.exists(): return str(out)
    from common.folds import inner_spatial_folds
    df, Y, NB, FB = _env(); fo = df[f'spatial_fold_r{r}'].values; blk = df[f'block_r{r}'].values
    tr = np.where(fo != f)[0]; te = np.where(fo == f)[0]
    pairs = inner_spatial_folds(tr, blk, SEED + 100 * f + r, k=5, repeats=1)
    P = np.zeros((len(te), 3, 2)); O = np.full((len(tr), 3, 2), np.nan); pos = {i: q for q, i in enumerate(tr)}
    for s, (name, spec) in enumerate(FEATURE_SETS.items()):
        for d in range(3):
            Xf, Xp, _ = FB.build(spec, tr, te, blk, d); P[:, d, s] = extratrees().fit(Xf, Y[tr, d]).predict_proba(Xp)[:, 1]
            for fit, val in pairs:
                Xf, Xp, _ = FB.build(spec, fit, val, blk, d); O[[pos[i] for i in val], d, s] = extratrees().fit(Xf, Y[fit, d]).predict_proba(Xp)[:, 1]
    E2.mkdir(parents=True, exist_ok=True); np.savez_compressed(out, test=P, oof=O, train_idx=tr, test_idx=te); return str(out)


# ------------------------------------------------------------------ E3: corrected 23-model library
def _lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05, min_child_samples=20, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, reg_lambda=1.0, n_jobs=1, verbose=-1, random_state=SEED)


def _svm():
    from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler; from sklearn.svm import SVC
    return make_pipeline(StandardScaler(), SVC(C=1.0, gamma='scale'))


def _lib():
    if 'G' not in _S: _S['G'] = load_library()
    return _S['G']


def _predict_elk(G, fit, pred, block):
    full = G['LIB']; elk = [j for j, m in enumerate(full) if m[2] == 'ELK']; G['LIB'] = [full[j] for j in elk]; G['C'] = len(elk)
    try: P = G['fit_predict_lib'](fit, pred, block)
    finally: G['LIB'] = full; G['C'] = len(full)
    Xf, Xp = G['featureset']('ELK', fit, pred, block); ex = np.zeros((len(pred), 3, 2)); Y = G['Y_ALL']
    for d in range(3):
        ex[:, d, 0] = _lgbm().fit(Xf, Y[fit, d]).predict_proba(Xp)[:, 1]; ex[:, d, 1] = _svm().fit(Xf, Y[fit, d]).decision_function(Xp)
    return elk, np.concatenate([P, ex], axis=2)


def e3_split(i):
    """Split i of the run-3 split list (development spatial splits are i = 5..19). Unaffected models (B, E, G feature sets)
    come from the Stage 4 cache; EL models from the audit A5 corrected-prior refit; ELK models, LightGBM and SVM are refitted
    here with the corrected prior AND deterministic kNN ties. Inner fold assignment = cached run-3 assignment."""
    out = E3 / f'split_{i:02d}.npz'
    if out.exists(): return str(out)
    G = _lib(); s = G['SPLITS'][i]; tr, te = s['train'], s['test']; C = np.load(CACHE23); fo = C[f'fold_{i}']
    Pt = C[f'test_{i}'].astype(float).copy(); Po = C[f'oof_{i}'].astype(float).copy()
    aff = [j for j, m in enumerate(G['LIB']) if m[2] in ('EL', 'ELK')] + [21, 22]; Z = np.load(A5_PARTS / f'split_{i:02d}.npz')
    Pt[:, :, aff] = Z['test']; Po[:, :, aff] = Z['oof']                       # corrected prior (EL and ELK, old ties)
    elk, T = _predict_elk(G, tr, te, s['block']); cols = elk + [21, 22]; Pt[:, :, cols] = T
    for k in range(5):
        q = fo == k; _, T = _predict_elk(G, tr[~q], tr[q], s['block']); tmp = Po[q]; tmp[:, :, cols] = T; Po[q] = tmp
    E3.mkdir(parents=True, exist_ok=True); np.savez_compressed(out, test=Pt, oof=Po, train_idx=tr, test_idx=te, rep=s['rep'], fold=s['fold']); return str(out)


# ------------------------------------------------------------------ E5: negative controls with corrected features
SIZE_BINS = [0, 1, 2, 4, 6, 9, 14, 10_000]


def permute_labels(Y, df, scheme, rng):
    if scheme == 'household': return Y[rng.permutation(len(Y))]
    loc = df['loc'].values; idx = {l: np.where(loc == l)[0] for l in np.unique(loc)}
    size = pd.Series({l: len(v) for l, v in idx.items()}); bins = pd.cut(size, SIZE_BINS); Yp = np.empty_like(Y)
    for _, members in size.groupby(bins, observed=True):
        rec = members.index.values; don = rng.permutation(rec)
        for r_, d_ in zip(rec, don):
            src = rng.permutation(idx[d_]); n = len(idx[r_])
            take = src[:n] if len(src) >= n else np.concatenate([src, rng.choice(idx[d_], n - len(src), replace=True)])
            Yp[idx[r_]] = Y[take]
    return Yp


def e5_part(scheme, perm):
    out = E5 / f'{scheme}_{perm:03d}.csv'
    if out.exists(): return str(out)
    df, Y, _, _ = _env()
    Yp = Y.copy() if perm < 0 else permute_labels(Y, df, scheme, np.random.default_rng(20261121 + (0 if scheme == 'household' else 50_000) + perm))
    _, _, _, FB = setup(Y_override=Yp); rows = []
    for r, f, tr, te, bc in dev_splits(df, reps=(1,)):
        for d, dn in enumerate(DIS):
            rec = {'scheme': scheme, 'perm': perm, 'fold': f, 'disease': dn, 'test_prevalence': Yp[te, d].mean()}
            for name, spec in FEATURE_SETS.items():
                Xf, Xp, _ = FB.build(spec, tr, te, df[bc].values, d); p = extratrees().fit(Xf, Yp[tr, d]).predict_proba(Xp)[:, 1]
                rec[f'{name}_ROC_AUC'] = auc(Yp[te, d], p); rec[f'{name}_PR_AUC'] = pr_auc(Yp[te, d], p)
            rows.append(rec)
    E5.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out, index=False); return str(out)


# ------------------------------------------------------------------ E6: proximity removal and block size
def _blocks(df, B, seed=20261121):
    rng = np.random.default_rng(seed + int(B)); res = {}
    for r in (1, 2, 3):
        ox, oy = (0.0, 0.0) if r == 1 else rng.uniform(0, B, 2)
        b = (np.floor((df.x_km + ox) / B).astype(int).astype(str) + '_' + np.floor((df.y_km + oy) / B).astype(int).astype(str)).values
        sizes = pd.Series(b).value_counts().to_dict(); load_ = np.zeros(5); asg = {}
        for g in rng.permutation(np.array(list(sizes), dtype=object)):
            f = int(np.argmin(load_)); asg[g] = f; load_[f] += sizes[g]
        res[r] = (b, pd.Series(b).map(asg).values)
    return res


def e6_job(cond, r, f):
    out = E6 / f'{cond}_r{r}_f{f}.csv'
    if out.exists(): return str(out)
    df, Y, NB, FB = _env(); loc = df['loc'].values
    if cond.startswith('blocks'):
        blk, fold = _blocks(df, float(cond[6:]))[r]
    else:
        blk, fold = df[f'block_r{r}'].values, df[f'spatial_fold_r{r}'].values
    te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]; te_locs = np.unique(loc[te]); dmin = NB.D[:, te_locs].min(1); removed = 0
    if cond.startswith(('buffer', 'matched')):
        Bkm = float(cond.split('_')[1]); near = dmin[loc[tr]] <= Bkm; removed = int(near.sum())
        if cond.startswith('buffer'): tr = tr[~near]
        else:
            rng = np.random.default_rng(20261121 + 1000 * r + 10 * f + int(Bkm)); drop, cnt = [], 0
            for l in rng.permutation(np.unique(loc[tr][~near])):
                if cnt >= removed: break
                drop.append(l); cnt += int((loc[tr] == l).sum())
            tr = tr[~np.isin(loc[tr], drop)]; removed = cnt
    tr_locs = np.unique(loc[tr]); nearest = NB.D[np.ix_(te_locs, tr_locs)].min(1); rows = []
    for d, dn in enumerate(DIS):
        rec = {'condition': cond, 'rep': r, 'fold': f, 'disease': dn, 'train_households': len(tr), 'train_locations': len(tr_locs), 'removed_households': removed,
               'train_positives': int(Y[tr, d].sum()), 'test_households': len(te), 'test_positives': int(Y[te, d].sum()),
               'median_nearest_training_location_km': float(np.median(nearest)), 'test_prevalence': Y[te, d].mean()}
        for name, spec in FEATURE_SETS.items():
            Xf, Xp, _ = FB.build(spec, tr, te, blk, d); p = extratrees().fit(Xf, Y[tr, d]).predict_proba(Xp)[:, 1]
            rec[f'{name}_ROC_AUC'] = auc(Y[te, d], p); rec[f'{name}_PR_AUC'] = pr_auc(Y[te, d], p)
        rows.append(rec)
    E6.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out, index=False); return str(out)


E6_CONDITIONS = ['base81'] + [f'{k}_{b}' for b in (25, 50, 100) for k in ('buffer', 'matched')] + ['blocks120', 'blocks160']


# ------------------------------------------------------------------ E7: forward temporal validation
def e7_temporal():
    out = E7 / 'forward_temporal_predictions_v1.npz'
    if out.exists(): return str(out)
    df, Y, NB, FB = _env(); tr = np.where(df.temporal_set.values == 'train')[0]; te = np.where(df.temporal_set.values == 'test')[0]
    P = np.zeros((len(te), 3, 2))
    for s, (name, spec) in enumerate(FEATURE_SETS.items()):
        for d in range(3):
            Xf, Xp, _ = FB.build(spec, tr, te, df.block_r1.values, d); P[:, d, s] = extratrees().fit(Xf, Y[tr, d]).predict_proba(Xp)[:, 1]
    E7.mkdir(parents=True, exist_ok=True); np.savez_compressed(out, test=P, train_idx=tr, test_idx=te); return str(out)


# ------------------------------------------------------------------ E8: grouped permutation importance (FS5, held-out folds)
def feature_groups(names, disease):
    g = {}
    for j, n in enumerate(names):
        if n in ('Altitude', 'Temperature', 'Precipitation', 'Relative_humidity', 'Wind_speed'): k = 'Environmental (5)'
        elif n in ('SL_temperature', 'DeMartonne', 'Temp_x_RH', 'Precip_x_RH', 'log_Precip', 'Wind_x_RH'): k = 'Climate indices (6)'
        elif n in ('Lat', 'Lon', 'LatPlusLon', 'LatMinusLon'): k = 'Coordinates (4)'
        elif n.startswith(('dens', 'knndist')): k = 'Sampling density (6)'
        elif int(n.split('_')[-1]) == disease: k = 'Own-disease rates (6)'
        else: k = 'Cross-disease rates (12)'
        g.setdefault(k, []).append(j)
    return g


def e8_fold(r, f, n_rep=10):
    out = E8 / f'r{r}_f{f}.csv'
    if out.exists(): return str(out)
    df, Y, NB, FB = _env(); fo = df[f'spatial_fold_r{r}'].values; blk = df[f'block_r{r}'].values; tr = np.where(fo != f)[0]; te = np.where(fo == f)[0]
    rng = np.random.default_rng(20261123 + 10 * r + f); rows = []
    for d, dn in enumerate(DIS):
        Xf, Xp, names = FB.build(FEATURE_SETS['FS5'], tr, te, blk, d); m = extratrees().fit(Xf, Y[tr, d]); base = auc(Y[te, d], m.predict_proba(Xp)[:, 1])
        for gname, cols in feature_groups(names, d).items():
            drops = []
            for _ in range(n_rep):
                Xq = Xp.copy(); perm = rng.permutation(len(Xq)); Xq[:, cols] = Xp[perm][:, cols]
                drops.append(base - auc(Y[te, d], m.predict_proba(Xq)[:, 1]))
            rows.append({'rep': r, 'fold': f, 'disease': dn, 'group': gname, 'baseline_ROC_AUC': base, 'mean_AUC_drop': float(np.mean(drops)), 'sd_AUC_drop': float(np.std(drops))})
    E8.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out, index=False); return str(out)
