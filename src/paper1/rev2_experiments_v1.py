"""Revision-2 experiments for IJIES Paper 1 (approved items A-G; development folds only; confirmatory partition untouched).
E11 progressive feature ablation FS1 -> FS5 (FS2-FS4 here; FS1/FS5 come from E2, same folds, inner folds and seeds).
E10b campaign-specific spatial CV with household-level predictions saved (re-run of E10 that keeps the predictions).
E12 SHAP values (TreeExplainer) and partial dependence for FS5 ExtraTrees, fitted on the training split only.
E13 area of applicability (Meyer & Pebesma 2021) at surveyed locations, FS1 and FS5 predictor spaces.
E14 campaign identification: how well FS1, neighbourhood-rate and FS5 features identify the survey year under spatial CV.
Every function writes a checkpoint under results/paper1_rev2_v1/ and returns at once if it exists."""
from pathlib import Path
import numpy as np, pandas as pd
from .core import setup, extratrees, FEATURE_SETS, _BASE, DIS, SEED, auc
from common.features import CLIMATE_NAMES
from common.folds import inner_spatial_folds

ROOT = Path(__file__).resolve().parents[2]; REV2 = ROOT / 'results' / 'paper1_rev2_v1'
E2 = ROOT / 'results' / 'paper1_v1' / 'E2_featuresets'
E10b, E11, E12, E13, E14 = (REV2 / n for n in ('E10b_campaign_predictions', 'E11_ablation', 'E12_shap_pdp', 'E13_aoa', 'E14_campaign_identification'))
ABLATION = {'FS2': {**_BASE, 'climate': CLIMATE_NAMES},                                                            # + 6 climate indices (11)
            'FS3': {**_BASE, 'climate': CLIMATE_NAMES, 'coordinates': True},                                        # + 4 coordinate terms (15)
            'FS4': {**_BASE, 'climate': CLIMATE_NAMES, 'coordinates': True, 'kernel_rates': True, 'knn_rates': True, 'cross_disease_rates': True}}  # + 18 rates (33)
RATES_ONLY = {**_BASE, 'climate': [], 'kernel_rates': True, 'knn_rates': True, 'cross_disease_rates': True}
_S = {}


def _env():
    if 'df' not in _S:
        df, Y, NB, FB = setup(); _S.update(df=df, Y=Y, NB=NB, FB=FB)
    return _S['df'], _S['Y'], _S['NB'], _S['FB']


def _split(r, f):
    df, Y, NB, FB = _env(); fo = df[f'spatial_fold_r{r}'].values; blk = df[f'block_r{r}'].values
    return np.where(fo != f)[0], np.where(fo == f)[0], blk


# ------------------------------------------------------------------ E11 ablation (identical protocol to E2)
def e11_split(r, f):
    out = E11 / f'r{r}_f{f}.npz'
    if out.exists(): return str(out)
    df, Y, NB, FB = _env(); tr, te, blk = _split(r, f)
    pairs = inner_spatial_folds(tr, blk, SEED + 100 * f + r, k=5, repeats=1); pos = {i: q for q, i in enumerate(tr)}
    P = np.zeros((len(te), 3, 3)); O = np.full((len(tr), 3, 3), np.nan); nfeat = {}
    for s, (name, spec) in enumerate(ABLATION.items()):
        for d in range(3):
            Xf, Xp, names = FB.build(spec, tr, te, blk, d); nfeat[name] = len(names); P[:, d, s] = extratrees().fit(Xf, Y[tr, d]).predict_proba(Xp)[:, 1]
            for fit, val in pairs:
                A, B_, _ = FB.build(spec, fit, val, blk, d); O[[pos[i] for i in val], d, s] = extratrees().fit(A, Y[fit, d]).predict_proba(B_)[:, 1]
    E11.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, test=P, oof=O, train_idx=tr, test_idx=te, sets=np.array(list(ABLATION)), n_features=np.array([nfeat[k] for k in ABLATION])); return str(out)


# ------------------------------------------------------------------ E10b campaign-specific predictions
def e10b_split(r, f):
    out = E10b / f'r{r}_f{f}.npz'
    if out.exists(): return str(out)
    df, Y, NB, FB = _env(); fo = df[f'spatial_fold_r{r}'].values; blk = df[f'block_r{r}'].values; yr = df.Year.values
    te_all = np.where(fo == f)[0]; P = np.full((len(te_all), 3, 2), np.nan); pos = {h: q for q, h in enumerate(te_all)}
    for t in np.unique(yr):
        tr = np.where((fo != f) & (yr == t))[0]; te = np.where((fo == f) & (yr == t))[0]
        if len(te) == 0 or len(tr) == 0: continue
        for d in range(3):
            if not (0 < Y[tr, d].sum() < len(tr)): continue
            for s, name in enumerate(('FS1', 'FS5')):
                Xf, Xp, _ = FB.build(FEATURE_SETS[name], tr, te, blk, d); P[[pos[h] for h in te], d, s] = extratrees().fit(Xf, Y[tr, d]).predict_proba(Xp)[:, 1]
    E10b.mkdir(parents=True, exist_ok=True); np.savez_compressed(out, campaign_test=P, test_idx=te_all); return str(out)


# ------------------------------------------------------------------ E12 SHAP and partial dependence (FS5 ExtraTrees)
def e12_split(r, f, n_grid=20):
    out = E12 / f'r{r}_f{f}.npz'
    if out.exists(): return str(out)
    import shap
    from sklearn.inspection import partial_dependence
    df, Y, NB, FB = _env(); tr, te, blk = _split(r, f); res = {}
    for d in range(3):
        Xf, Xp, names = FB.build(FEATURE_SETS['FS5'], tr, te, blk, d); m = extratrees().fit(Xf, Y[tr, d])
        sv = shap.TreeExplainer(m).shap_values(Xp, check_additivity=False); sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
        res[f'shap_{d}'] = sv.astype(np.float32); res[f'X_{d}'] = Xp.astype(np.float32); res[f'imp_{d}'] = m.feature_importances_
        grids, pds = [], []
        for j in range(Xf.shape[1]):                                  # partial dependence on the training data only (no test labels)
            pdr = partial_dependence(m, Xf, [j], grid_resolution=n_grid, kind='average', percentiles=(0.05, 0.95))
            g = np.full(n_grid, np.nan); p = np.full(n_grid, np.nan); gv = pdr['grid_values'][0]; g[:len(gv)] = gv; p[:len(gv)] = pdr['average'][0]
            grids.append(g); pds.append(p)
        res[f'pd_grid_{d}'] = np.array(grids); res[f'pd_avg_{d}'] = np.array(pds)
    E12.mkdir(parents=True, exist_ok=True); np.savez_compressed(out, names=np.array(names), test_idx=te, **res); return str(out)


# ------------------------------------------------------------------ E13 area of applicability (dissimilarity index)
def _di(Xtr, Xq, w, sd, mu):
    A = (Xtr - mu) / sd * w; Q = (Xq - mu) / sd * w
    d2 = (Q ** 2).sum(1)[:, None] + (A ** 2).sum(1)[None, :] - 2 * Q @ A.T
    return np.sqrt(np.maximum(d2, 0))


def e13_split(r, f):
    """DI = weighted standardised distance to the nearest training point / mean pairwise training distance. Weights are the
    impurity importances of the model fitted on the training split; standardisation uses training means and SDs. The AOA
    threshold is the upper whisker (Q3 + 1.5 IQR, capped at the maximum) of training DIs computed across inner spatial folds."""
    out = E13 / f'r{r}_f{f}.csv'
    if out.exists(): return str(out)
    df, Y, NB, FB = _env(); tr, te, blk = _split(r, f); inner = inner_spatial_folds(tr, blk, SEED + 100 * f + r, k=5, repeats=1); rows = []
    for name in ('FS1', 'FS5'):
        for d, dn in enumerate(DIS):
            Xf, Xp, _ = FB.build(FEATURE_SETS[name], tr, te, blk, d); m = extratrees().fit(Xf, Y[tr, d]); w = m.feature_importances_
            mu, sd = Xf.mean(0), Xf.std(0); sd[sd == 0] = 1.0
            Dtt = _di(Xf, Xf, w, sd, mu); dbar = Dtt[np.triu_indices(len(Xf), 1)].mean()
            di_tr = np.empty(len(tr)); pos = {i: q for q, i in enumerate(tr)}
            for fit, val in inner:
                iv = [pos[i] for i in val]; ifit = [pos[i] for i in fit]; di_tr[iv] = Dtt[np.ix_(iv, ifit)].min(1) / dbar
            q1, q3 = np.percentile(di_tr, [25, 75]); thr = min(q3 + 1.5 * (q3 - q1), di_tr.max())
            di_te = _di(Xf, Xp, w, sd, mu).min(1) / dbar
            for h, v in zip(te, di_te): rows.append({'rep': r, 'fold': f, 'features': name, 'disease': dn, 'household': int(h), 'DI': float(v), 'threshold': float(thr), 'inside_AOA': bool(v <= thr)})
    E13.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out, index=False); return str(out)


# ------------------------------------------------------------------ E14 campaign identification
def e14_split(r, f):
    out = E14 / f'r{r}_f{f}.csv'
    if out.exists(): return str(out)
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import roc_auc_score
    df, Y, NB, FB = _env(); tr, te, blk = _split(r, f); yr = df.Year.values; rows = []
    for name, spec in (('FS1', FEATURE_SETS['FS1']), ('Neighbourhood rates only', RATES_ONLY), ('FS5', FEATURE_SETS['FS5'])):
        Xf, Xp, _ = FB.build(spec, tr, te, blk, 0)
        m = ExtraTreesClassifier(300, min_samples_leaf=10, max_features='sqrt', n_jobs=1, random_state=SEED).fit(Xf, yr[tr]); P = m.predict_proba(Xp)
        present = [c for c in m.classes_ if (yr[te] == c).any() and (yr[te] != c).any()]
        per = {int(c): roc_auc_score(yr[te] == c, P[:, list(m.classes_).index(c)]) for c in present}
        rows.append({'rep': r, 'fold': f, 'features': name, 'macro_AUC_year': float(np.mean(list(per.values()))), 'years_in_test': len(present),
                     'accuracy': float((m.classes_[P.argmax(1)] == yr[te]).mean()), **{f'AUC_{k}': v for k, v in per.items()}})
    E14.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out, index=False); return str(out)


if __name__ == '__main__':
    import sys, time
    t0 = time.time(); jobs = sys.argv[1:] or ['E11', 'E10b', 'E14', 'E13', 'E12']
    for job in jobs:
        for r in (1, 2, 3):
            for f in range(5):
                if job == 'E12' and r != 1: continue
                {'E11': e11_split, 'E10b': e10b_split, 'E12': e12_split, 'E13': e13_split, 'E14': e14_split}[job](r, f)
                print(f'{job} r{r} f{f} ({(time.time() - t0) / 60:.1f} min)', flush=True)
    print('ALL DONE', flush=True)
