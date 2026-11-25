"""Feature-leakage audit A7 (temporal availability) + A9 (training/test feature distributions), 2026-11-21.
Development partitions only (3 x 5 folds, 81 km blocks). EXPLORATORY.
A7  (1) For every query household, the share of the neighbourhood information (150 km kernel weight, and households
        among the 15 nearest locations) that comes from surveys in LATER years than the household's own survey.
    (2) Scenario B features: rates rebuilt per survey year y from training households surveyed in year <= y only
        (concurrent + historical surveillance; no later reports), then FS1 vs FS5 (pooled years, Scenario A) vs
        FS5-temporal, ExtraTrees-300, all 3 diseases.
A9  Per outer fold: distribution of every FS5 rate feature in training vs test rows, and per-row information
    conditions (effective number of contributing locations, prior share, fallback, nearest contributing location,
    contributing households/locations within 150 km). Run from the project root."""
import os, sys, time, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
SRC = Path(__file__).resolve().parent; sys.path.insert(0, str(SRC)); os.environ['PYTHONPATH'] = str(SRC); ROOT = SRC.parent
warnings.filterwarnings('ignore')
OUT7 = ROOT / 'results/audit_v1/A7'; OUT9 = ROOT / 'results/audit_v1/A9'; OUT7.mkdir(parents=True, exist_ok=True); OUT9.mkdir(parents=True, exist_ok=True)
RATES = dict(bandwidths=(50.0, 150.0, 400.0), ks=(5, 15, 40), prior=1.0, idw_power=None, density=True, diseases=(0, 1, 2))
_S = {}


def setup():
    if not _S:
        from common.data import load, labels
        from common.features import Neighbourhood, climate_coords, CLIMATE_NAMES, COORD_NAMES
        df = load(ROOT / 'data/raw'); Y = labels(df); C = climate_coords(df)
        _S.update(df=df, Y=Y, NB=Neighbourhood(df, Y), CL=C[CLIMATE_NAMES + COORD_NAMES].values, FS1=C[['Altitude', 'Temperature', 'Precipitation', 'Relative_humidity', 'Wind_speed']].values)
    return _S


def a7_split(r, f):
    from common.data import SEED
    from common.fitness import auc_avg
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import average_precision_score
    S = setup(); df, Y, NB = S['df'], S['Y'], S['NB']; loc = df['loc'].values; yr = df.Year.values
    fold = df[f'spatial_fold_r{r}'].values; blk = df[f'block_r{r}'].values; te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
    bl = NB.block_of_loc(blk); years = np.sort(np.unique(yr))
    # (1) share of neighbourhood information from later surveys
    nly = np.zeros((462, len(years)))
    for i in tr: nly[loc[i], np.searchsorted(years, yr[i])] += 1
    n = nly.sum(1); K = np.exp(-NB.D ** 2 / (2 * 150.0 ** 2)); avail = (bl[:, None] != bl[None, :]) & (n[None, :] > 0); Kx = np.where(avail, K, 0.0)
    Dm = np.where(avail, NB.D, np.inf); nn15 = np.argsort(Dm, axis=1, kind='stable')[:, :15]
    share = []
    for part, rows in (('train', tr), ('test', te)):
        kw = Kx[loc[rows]] @ nly; yi = np.searchsorted(years, yr[rows]); later = np.array([kw[q, yi[q] + 1:].sum() for q in range(len(rows))]) / np.maximum(kw.sum(1), 1e-300)
        nk = nly[nn15[loc[rows]]].sum(1); later_k = np.array([nk[q, yi[q] + 1:].sum() for q in range(len(rows))]) / np.maximum(nk.sum(1), 1e-300)
        for y_ in years:
            m = yr[rows] == y_
            if m.any(): share.append({'rep': r, 'fold': f, 'rows': part, 'survey_year': int(y_), 'households': int(m.sum()),
                                      'later_share_kernel150_weight': float(later[m].mean()), 'later_share_nearest15_households': float(later_k[m].mean())})
        share.append({'rep': r, 'fold': f, 'rows': part, 'survey_year': 'all', 'households': len(rows), 'later_share_kernel150_weight': float(later.mean()),
                      'later_share_nearest15_households': float(later_k.mean())})
    # (2) Scenario B features: per year y, fit rows with Year <= y
    Ftr = np.zeros((len(tr), 24)); Fte = np.zeros((len(te), 24)); ptr = {i: q for q, i in enumerate(tr)}; pte = {i: q for q, i in enumerate(te)}
    for y_ in years:
        fit_y = tr[yr[tr] <= y_]; qtr = tr[yr[tr] == y_]; qte = te[yr[te] == y_]
        if len(qtr): _, F, _ = NB.rates(fit_y, qtr, blk, **RATES); Ftr[[ptr[i] for i in qtr]] = F
        if len(qte): _, F, _ = NB.rates(fit_y, qte, blk, **RATES); Fte[[pte[i] for i in qte]] = F
    Fp_tr, Fp_te, _ = NB.rates(tr, te, blk, **RATES)
    X = {'FS1': (S['FS1'][tr], S['FS1'][te]), 'FS5 pooled years (Scenario A)': (np.column_stack([S['CL'][tr], Fp_tr]), np.column_stack([S['CL'][te], Fp_te])),
         'FS5 year <= own year (Scenario B)': (np.column_stack([S['CL'][tr], Ftr]), np.column_stack([S['CL'][te], Fte]))}
    res = []
    for d, dn in enumerate(['CBPP', 'FMD', 'LSD']):
        rec = {'rep': r, 'fold': f, 'disease': dn}
        for name, (a, b) in X.items():
            p = ExtraTreesClassifier(300, min_samples_leaf=10, max_features='sqrt', n_jobs=1, random_state=SEED).fit(a, Y[tr, d]).predict_proba(b)[:, 1]
            rec[f'{name} ROC_AUC'] = auc_avg(Y[te, d], p); rec[f'{name} PR_AUC'] = average_precision_score(Y[te, d], p)
        res.append(rec)
    return share, res


def a9_split(r, f):
    S = setup(); df, Y, NB = S['df'], S['Y'], S['NB']; loc = df['loc'].values
    fold = df[f'spatial_fold_r{r}'].values; blk = df[f'block_r{r}'].values; te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
    Ftr, Fte, names = NB.rates(tr, te, blk, **RATES); bl = NB.block_of_loc(blk); n = np.bincount(loc[tr], minlength=462).astype(float)
    avail = (bl[:, None] != bl[None, :]) & (n[None, :] > 0)
    K50 = np.where(avail, np.exp(-NB.D ** 2 / (2 * 50.0 ** 2)), 0.0); K150 = np.where(avail, np.exp(-NB.D ** 2 / (2 * 150.0 ** 2)), 0.0)
    w = K150 * n[None, :]; neff = w.sum(1) ** 2 / np.maximum((w ** 2).sum(1), 1e-300); prior_share50 = 1.0 / (K50 @ n + 1.0)
    nearest = np.where(avail, NB.D, np.inf).min(1); within = avail & (NB.D <= 150)
    info = {'effective_contributing_locations_150km': neff, 'prior_share_of_50km_rate': prior_share50, 'fallback_50km (prior share > 0.5)': (prior_share50 > 0.5).astype(float),
            'nearest_contributing_location_km': nearest, 'contributing_locations_within_150km': within.sum(1).astype(float),
            'contributing_households_within_150km': (within * n[None, :]).sum(1), 'contributing_locations_total': avail.sum(1).astype(float)}
    rows = []
    for part, idx, F in (('train', tr, Ftr), ('test', te, Fte)):
        for j, nm in enumerate(names):
            v = F[:, j]; rows.append({'rep': r, 'fold': f, 'rows': part, 'quantity': nm, 'mean': v.mean(), 'sd': v.std(), 'median': np.median(v), 'q05': np.quantile(v, 0.05),
                                      'q25': np.quantile(v, 0.25), 'q75': np.quantile(v, 0.75), 'q95': np.quantile(v, 0.95), 'missing': int(np.isnan(v).sum())})
        for nm, arr in info.items():
            v = arr[loc[idx]]; rows.append({'rep': r, 'fold': f, 'rows': part, 'quantity': nm, 'mean': v.mean(), 'sd': v.std(), 'median': np.median(v), 'q05': np.quantile(v, 0.05),
                                            'q25': np.quantile(v, 0.25), 'q75': np.quantile(v, 0.75), 'q95': np.quantile(v, 0.95), 'missing': int(np.isnan(v).sum())})
    return rows


if __name__ == '__main__':
    t0 = time.time(); splits = [(r, f) for r in (1, 2, 3) for f in range(5)]
    A9 = pd.DataFrame([x for rows in Parallel(n_jobs=2)(delayed(a9_split)(r, f) for r, f in splits) for x in rows]); A9.to_csv(OUT9 / 'A9_fold_distributions_v1.csv', index=False)
    W = A9.groupby(['quantity', 'rows'])[['mean', 'sd', 'median', 'q05', 'q95', 'missing']].mean().unstack('rows')
    W.columns = [f'{a}_{b}' for a, b in W.columns]; W['test_minus_train_mean'] = W.mean_test - W.mean_train
    W['standardised_difference'] = W.test_minus_train_mean / np.sqrt((W.sd_train ** 2 + W.sd_test ** 2) / 2); W.reset_index().to_csv(OUT9 / 'A9_train_vs_test_summary_v1.csv', index=False)
    print('A9 done', round((time.time() - t0) / 60, 1), 'min', flush=True)
    res = Parallel(n_jobs=2)(delayed(a7_split)(r, f) for r, f in splits)
    SH = pd.DataFrame([x for s, _ in res for x in s]); SH.to_csv(OUT7 / 'A7_later_survey_share_by_fold_v1.csv', index=False)
    SHs = SH.groupby(['rows', 'survey_year'])[['households', 'later_share_kernel150_weight', 'later_share_nearest15_households']].mean().reset_index(); SHs.to_csv(OUT7 / 'A7_later_survey_share_summary_v1.csv', index=False)
    R7 = pd.DataFrame([x for _, rr in res for x in rr]); R7.to_csv(OUT7 / 'A7_scenarioB_fold_metrics_v1.csv', index=False)
    a, b, c = 'FS1 ROC_AUC', 'FS5 pooled years (Scenario A) ROC_AUC', 'FS5 year <= own year (Scenario B) ROC_AUC'
    R7['gain_A'] = R7[b] - R7[a]; R7['gain_B'] = R7[c] - R7[a]; R7['B_minus_A'] = R7[c] - R7[b]
    S7 = R7.groupby('disease').agg(FS1=(a, 'mean'), FS5_scenarioA=(b, 'mean'), FS5_scenarioB=(c, 'mean'), gain_A=('gain_A', 'mean'), gain_B=('gain_B', 'mean'),
                                   gain_B_SE=('gain_B', lambda s: s.std(ddof=1) / np.sqrt(len(s))), folds_gain_B_positive=('gain_B', lambda s: f'{(s > 0).sum()}/15'),
                                   B_minus_A=('B_minus_A', 'mean')).reset_index(); S7.to_csv(OUT7 / 'A7_scenarioB_summary_v1.csv', index=False)
    pd.set_option('display.width', 250)
    print(SHs.round(3).to_string(index=False)); print(S7.round(4).to_string(index=False))
    print(W[['mean_train', 'mean_test', 'standardised_difference']].round(3).to_string()); print(f'A7/A9 done in {(time.time() - t0) / 60:.1f} min')
