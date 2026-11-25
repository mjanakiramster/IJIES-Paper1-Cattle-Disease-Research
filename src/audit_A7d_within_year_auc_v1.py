"""Audit A7d (2026-11-21), EXPLORATORY: survey-year (campaign) confounding check.
Prevalence differs strongly between survey years, and year-restricted features (A7 Scenario B, A7c other-years) change
their fit population with the query's year, so they can encode the survey year itself. This script (1) reports
prevalence by survey year, and (2) re-scores FS1, pooled-year FS5 (the study's features) and the A7c other-years FS5 with
a WITHIN-YEAR ROC-AUC: only positive/negative pairs from the same survey year are compared (pair-weighted over years).
A gain that survives within years is not explained by between-campaign prevalence differences. Dev 3 x 5 folds."""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
SRC = Path(__file__).resolve().parent; sys.path.insert(0, str(SRC)); os.environ['PYTHONPATH'] = str(SRC); ROOT = SRC.parent
warnings.filterwarnings('ignore'); OUT = ROOT / 'results/audit_v1/A7'
RATES = dict(bandwidths=(50.0, 150.0, 400.0), ks=(5, 15, 40), prior=1.0, idw_power=None, density=True, diseases=(0, 1, 2))
def within_year_auc(y, s, yr):
    num = den = 0.0
    for v in np.unique(yr):
        m = yr == v; yy, ss = y[m], s[m]; n1 = yy.sum(); n0 = len(yy) - n1
        if n1 == 0 or n0 == 0: continue
        r = pd.Series(ss).rank().values; num += r[yy == 1].sum() - n1 * (n1 + 1) / 2; den += n1 * n0
    return num / den
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
        for k, (a, b) in X.items():
            p = ExtraTreesClassifier(300, min_samples_leaf=10, max_features='sqrt', n_jobs=1, random_state=SEED).fit(a, Y[tr, d]).predict_proba(b)[:, 1]
            out.append({'rep': r, 'fold': f, 'disease': dn, 'features': k, 'ROC_AUC': auc_avg(Y[te, d], p), 'within_year_ROC_AUC': within_year_auc(Y[te, d], p, yr[te])})
    return out
if __name__ == '__main__':
    from common.data import load, labels
    df = load(ROOT / 'data/raw'); Y = labels(df)
    prev = pd.DataFrame(Y, columns=['CBPP', 'FMD', 'LSD']).groupby(df.Year.values).mean().round(3); prev['households'] = df.Year.value_counts().sort_index().values
    prev.to_csv(OUT / 'A7d_prevalence_by_survey_year_v1.csv'); print(prev.to_string())
    R = pd.DataFrame([x for o in Parallel(n_jobs=2)(delayed(split)(r, f) for r in (1, 2, 3) for f in range(5)) for x in o]); R.to_csv(OUT / 'A7d_within_year_fold_metrics_v1.csv', index=False)
    W = R.pivot_table(index=['rep', 'fold', 'disease'], columns='features', values=['ROC_AUC', 'within_year_ROC_AUC'])
    rows = []
    for d in ['CBPP', 'FMD', 'LSD']:
        g = W.xs(d, level='disease'); rec = {'disease': d}
        for m in ('ROC_AUC', 'within_year_ROC_AUC'):
            for k in ('FS1', 'FS5 pooled years', 'FS5 other years only'): rec[f'{m} {k}'] = g[(m, k)].mean()
            gp = g[(m, 'FS5 pooled years')] - g[(m, 'FS1')]; rec[f'{m} gain pooled'] = gp.mean(); rec[f'{m} gain pooled folds>0'] = f'{(gp > 0).sum()}/15'
            rec[f'{m} gain pooled SE'] = gp.std(ddof=1) / np.sqrt(15); rec[f'{m} gain other-years'] = (g[(m, 'FS5 other years only')] - g[(m, 'FS1')]).mean()
        rows.append(rec)
    S = pd.DataFrame(rows); S.to_csv(OUT / 'A7d_within_year_summary_v1.csv', index=False); pd.set_option('display.width', 300); print(S.round(4).T.to_string())
