"""Revision experiments for IJIES Paper 1 (corrected code; development data only; confirmatory partition untouched).
E9  random 5-fold CV with the corrected code (FS1/FS5 ExtraTrees and the 23-model library), so that random and spatial CV
    come from the same pipeline (resolves the Section 4.1 / Table 4 mismatch of manuscript v4).
E10 campaign-specific spatial CV: for each survey year, models trained only on that year's training households and tested
    on that year's test households, compared on the same households with the pooled (all-years) models from E2.
Each function writes a checkpoint under results/paper1_rev_v1/ and returns at once if it exists."""
from pathlib import Path
import numpy as np, pandas as pd
from .core import setup, extratrees, load_library, FEATURE_SETS, DIS, SEED, auc, pr_auc
from .experiments import CACHE23, _lgbm, _svm

ROOT = Path(__file__).resolve().parents[2]; REV = ROOT / 'results' / 'paper1_rev_v1'
E9 = REV / 'E9_random_cv'; E10 = REV / 'E10_campaign_cv'; E2 = ROOT / 'results' / 'paper1_v1' / 'E2_featuresets'
_S = {}


def _env():
    if 'df' not in _S:
        df, Y, NB, FB = setup(); _S.update(df=df, Y=Y, NB=NB, FB=FB)
    return _S['df'], _S['Y'], _S['NB'], _S['FB']


def _lib():
    if 'G' not in _S: _S['G'] = load_library()
    return _S['G']


def _predict_affected(G, fit, pred, block):
    """Refit every library model that uses outcome-derived features (EL, ELK) plus LightGBM and SVM, with the corrected code."""
    full = G['LIB']; aff = [j for j, m in enumerate(full) if m[2] in ('EL', 'ELK')]; G['LIB'] = [full[j] for j in aff]; G['C'] = len(aff)
    try: P = G['fit_predict_lib'](fit, pred, block)
    finally: G['LIB'] = full; G['C'] = len(full)
    Xf, Xp = G['featureset']('ELK', fit, pred, block); ex = np.zeros((len(pred), 3, 2)); Y = G['Y_ALL']
    for d in range(3):
        ex[:, d, 0] = _lgbm().fit(Xf, Y[fit, d]).predict_proba(Xp)[:, 1]; ex[:, d, 1] = _svm().fit(Xf, Y[fit, d]).decision_function(Xp)
    return aff + [21, 22], np.concatenate([P, ex], axis=2)


def e9_split(i):
    """Random-CV split i (run-3 split list, i = 0..4; own-block exclusion uses block_r1, as in run 3 and Stage 4)."""
    out = E9 / f'random_{i}.npz'
    if out.exists(): return str(out)
    G = _lib(); s = G['SPLITS'][i]; assert s['scheme'] == 'random'; tr, te, blk_col = s['train'], s['test'], s['block']
    C = np.load(CACHE23); fo = C[f'fold_{i}']
    Pt = C[f'test_{i}'].astype(float).copy(); Po = C[f'oof_{i}'].astype(float).copy()
    cols, T = _predict_affected(G, tr, te, blk_col); Pt[:, :, cols] = T
    for k in range(5):
        q = fo == k; _, T = _predict_affected(G, tr[~q], tr[q], blk_col); tmp = Po[q]; tmp[:, :, cols] = T; Po[q] = tmp
    df, Y, NB, FB = _env(); blk = df[blk_col].values
    PF = np.zeros((len(te), 3, 2)); OF = np.full((len(tr), 3, 2), np.nan)
    for sidx, (name, spec) in enumerate(FEATURE_SETS.items()):
        for d in range(3):
            Xf, Xp, _ = FB.build(spec, tr, te, blk, d); PF[:, d, sidx] = extratrees().fit(Xf, Y[tr, d]).predict_proba(Xp)[:, 1]
            for k in range(5):
                q = fo == k; Xf, Xp, _ = FB.build(spec, tr[~q], tr[q], blk, d); OF[q, d, sidx] = extratrees().fit(Xf, Y[tr[~q], d]).predict_proba(Xp)[:, 1]
    E9.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, lib_test=Pt, lib_oof=Po, fs_test=PF, fs_oof=OF, train_idx=tr, test_idx=te, inner_fold=fo); return str(out)


CAMPAIGN_YEARS = (2012, 2014, 2015, 2018, 2019)


def e10_split(r, f):
    """Campaign-specific models for development split (r, f); compared with the pooled E2 models on the same test households."""
    out = E10 / f'r{r}_f{f}.csv'
    if out.exists(): return str(out)
    df, Y, NB, FB = _env(); fo = df[f'spatial_fold_r{r}'].values; blk = df[f'block_r{r}'].values; yr = df.Year.values
    Z = np.load(E2 / f'r{r}_f{f}.npz'); te_all = Z['test_idx']; pooled = Z['test']; pos = {h: q for q, h in enumerate(te_all)}; rows = []
    for t in CAMPAIGN_YEARS:
        tr = np.where((fo != f) & (yr == t))[0]; te = np.where((fo == f) & (yr == t))[0]
        for d, dn in enumerate(DIS):
            yt = Y[te, d]; rec = {'rep': r, 'fold': f, 'year': t, 'disease': dn, 'train_households': len(tr), 'train_positives': int(Y[tr, d].sum()),
                                  'test_households': len(te), 'test_positives': int(yt.sum())}
            ok = len(te) > 0 and 0 < yt.sum() < len(te) and 0 < Y[tr, d].sum() < len(tr)
            for sidx, name in enumerate(FEATURE_SETS):
                if ok:
                    Xf, Xp, _ = FB.build(FEATURE_SETS[name], tr, te, blk, d); p = extratrees().fit(Xf, Y[tr, d]).predict_proba(Xp)[:, 1]
                    rec[f'campaign_{name}_ROC_AUC'] = auc(yt, p); rec[f'pooled_{name}_ROC_AUC'] = auc(yt, pooled[[pos[h] for h in te], d, sidx])
                else:
                    rec[f'campaign_{name}_ROC_AUC'] = np.nan; rec[f'pooled_{name}_ROC_AUC'] = np.nan
            rows.append(rec)
    E10.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out, index=False); return str(out)


if __name__ == '__main__':
    import sys, time
    t0 = time.time()
    for r in (1, 2, 3):
        for f in range(5): e10_split(r, f); print(f'E10 r{r} f{f} ({(time.time() - t0) / 60:.1f} min)', flush=True)
    for i in range(5): e9_split(i); print(f'E9 random {i} ({(time.time() - t0) / 60:.1f} min)', flush=True)
    print('ALL DONE', flush=True)
