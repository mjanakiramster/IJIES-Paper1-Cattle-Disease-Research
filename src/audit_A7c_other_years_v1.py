"""Audit A7c (2026-11-21), EXPLORATORY: is the engineered-feature gain driven by SAME-survey-year (campaign) neighbours?
Rates for a household surveyed in year y are rebuilt from training households surveyed in OTHER years only (year != y),
own 81 km block still excluded. Compared with FS1 and the standard pooled-year FS5 on development 3 x 5 folds
(ExtraTrees-300). If the gain survives, it reflects spatial structure that persists across survey years; if it vanishes,
it reflects campaign-specific (same-year) clustering. Run from the project root."""
import os, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
SRC = Path(__file__).resolve().parent; sys.path.insert(0, str(SRC)); os.environ['PYTHONPATH'] = str(SRC); ROOT = SRC.parent
warnings.filterwarnings('ignore'); OUT = ROOT / 'results/audit_v1/A7'
RATES = dict(bandwidths=(50.0, 150.0, 400.0), ks=(5, 15, 40), prior=1.0, idw_power=None, density=True, diseases=(0, 1, 2))
def split(r, f):
    from common.data import load, labels, SEED, ENV
    from common.features import Neighbourhood, climate_coords, CLIMATE_NAMES, COORD_NAMES
    from common.fitness import auc_avg
    from sklearn.ensemble import ExtraTreesClassifier
    df = load(ROOT / 'data/raw'); Y = labels(df); NB = Neighbourhood(df, Y); C = climate_coords(df); CL = C[CLIMATE_NAMES + COORD_NAMES].values; F1 = C[ENV].values
    yr = df.Year.values; fold = df[f'spatial_fold_r{r}'].values; blk = df[f'block_r{r}'].values; te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
    Ftr = np.zeros((len(tr), 24)); Fte = np.zeros((len(te), 24)); ptr = {i: q for q, i in enumerate(tr)}; pte = {i: q for q, i in enumerate(te)}
    for y_ in np.unique(yr):
        fit_y = tr[yr[tr] != y_]; qtr = tr[yr[tr] == y_]; qte = te[yr[te] == y_]
        if len(qtr): Ftr[[ptr[i] for i in qtr]] = NB.rates(fit_y, qtr, blk, **RATES)[1]
        if len(qte): Fte[[pte[i] for i in qte]] = NB.rates(fit_y, qte, blk, **RATES)[1]
    Pa, Pb, _ = NB.rates(tr, te, blk, **RATES)
    X = {'FS1': (F1[tr], F1[te]), 'FS5 pooled years': (np.column_stack([CL[tr], Pa]), np.column_stack([CL[te], Pb])),
         'FS5 other years only': (np.column_stack([CL[tr], Ftr]), np.column_stack([CL[te], Fte]))}
    out = []
    for d, dn in enumerate(['CBPP', 'FMD', 'LSD']):
        rec = {'rep': r, 'fold': f, 'disease': dn}
        for k, (a, b) in X.items():
            rec[k] = auc_avg(Y[te, d], ExtraTreesClassifier(300, min_samples_leaf=10, max_features='sqrt', n_jobs=1, random_state=SEED).fit(a, Y[tr, d]).predict_proba(b)[:, 1])
        out.append(rec)
    return out
if __name__ == '__main__':
    R = pd.DataFrame([x for o in Parallel(n_jobs=2)(delayed(split)(r, f) for r in (1, 2, 3) for f in range(5)) for x in o]); R.to_csv(OUT / 'A7c_other_years_fold_metrics_v1.csv', index=False)
    R['gain_pooled'] = R['FS5 pooled years'] - R.FS1; R['gain_other_years'] = R['FS5 other years only'] - R.FS1
    S = R.groupby('disease').agg(FS1=('FS1', 'mean'), FS5_pooled=('FS5 pooled years', 'mean'), FS5_other_years=('FS5 other years only', 'mean'), gain_pooled=('gain_pooled', 'mean'),
                                 gain_other_years=('gain_other_years', 'mean'), gain_other_years_SE=('gain_other_years', lambda s: s.std(ddof=1) / np.sqrt(len(s))),
                                 folds_other_years_gain_positive=('gain_other_years', lambda s: f'{(s > 0).sum()}/15')).reset_index()
    S.to_csv(OUT / 'A7c_other_years_summary_v1.csv', index=False); print(S.round(4).to_string(index=False))
