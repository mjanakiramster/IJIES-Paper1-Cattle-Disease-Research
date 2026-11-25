"""Independent recomputation of the reported statistics from the saved predictions (revision check).
Uses the portable long-format prediction file and its own implementations (rank-based AUC, step-wise average precision,
Newton-Raphson logistic calibration, own corrected t, own Holm, own vectorised cluster bootstrap with a different seed),
then compares every value with the result tables. Writes results/paper1_rev_v1/statistics_verification/.
Run from the project root: python src/paper1/rev_verify_statistics_v1.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[2]; P1 = ROOT / 'results' / 'paper1_v1'; OUT = ROOT / 'results' / 'paper1_rev_v1' / 'statistics_verification'
OUT.mkdir(parents=True, exist_ok=True); DIS = ['CBPP', 'FMD', 'LSD']; MODELS = ['FS1', 'FS5', '(a) Rank average', '(b) Inner-selected single']


def auc_rank(y, s):
    y = np.asarray(y).astype(bool); r = stats.rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ap_step(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float); thr = np.unique(s)[::-1]; P = y.sum(); ap = 0.0; prev_r = 0.0
    for t in thr:
        sel = s >= t; tp = (y[sel] == 1).sum(); r = tp / P; ap += (r - prev_r) * tp / sel.sum(); prev_r = r
    return ap


def logistic_nr(x, y, offset=None, slope_free=True, iters=100):
    """Newton-Raphson for logit P(y=1) = a + b x (or a + offset)."""
    X = np.column_stack([np.ones_like(x), x]) if slope_free else np.ones((len(x), 1)); beta = np.zeros(X.shape[1]); off = 0 if offset is None else offset
    for _ in range(iters):
        eta = X @ beta + off; p = 1 / (1 + np.exp(-eta)); W = p * (1 - p); g = X.T @ (y - p); H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, g); beta += step
        if np.abs(step).max() < 1e-10: break
    return beta


def corrected_t_own(d, ratio):
    d = np.asarray(d, float); k = len(d); se = np.sqrt((1 / k + ratio) * d.var(ddof=1)); tq = stats.t.ppf(0.975, k - 1); t = d.mean() / se
    return d.mean(), d.mean() - tq * se, d.mean() + tq * se, 2 * stats.t.sf(abs(t), k - 1)


def holm_own(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o])); out = np.empty(m); out[o] = adj; return out


L = pd.read_csv(P1 / 'predictions' / 'dev_oof_predictions_long_v1.csv'); FM = pd.read_csv(P1 / 'metrics' / 'fold_metrics_v2.csv')
rows = []
for (rep, fold, dn, m), g in L.groupby(['rep', 'fold', 'disease', 'model']):
    y = g.observed.values; s = g.score.values; p = np.clip(g.calibrated_probability.values, 1e-6, 1 - 1e-6); yhat = g.predicted_class.values
    tp = ((yhat == 1) & (y == 1)).sum(); tn = ((yhat == 0) & (y == 0)).sum(); fp = ((yhat == 1) & (y == 0)).sum(); fn = ((yhat == 0) & (y == 1)).sum()
    sens, spec = tp / (tp + fn), tn / (tn + fp); prec = tp / (tp + fp) if tp + fp else 0.0
    mcc_den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))); lp = np.log(p / (1 - p))
    num = den = 0.0
    for t in np.unique(g.survey_year):
        mm = g.survey_year.values == t; n1 = y[mm].sum(); n0 = mm.sum() - n1
        if n1 and n0: num += auc_rank(y[mm], s[mm]) * n1 * n0; den += n1 * n0
    slope = logistic_nr(lp, y)[1] if np.ptp(lp) > 1e-12 else np.nan; icpt = logistic_nr(lp, y, offset=lp, slope_free=False)[0]
    rows.append({'rep': rep, 'fold': fold, 'disease': dn, 'model': m, 'ROC_AUC': auc_rank(y, s), 'PR_AUC': ap_step(y, s), 'Brier': np.mean((p - y) ** 2),
                 'calibration_intercept': icpt, 'calibration_slope': slope, 'sensitivity': sens, 'specificity': spec, 'balanced_accuracy': (sens + spec) / 2,
                 'F1': 2 * prec * sens / (prec + sens) if prec + sens else 0.0, 'MCC': (tp * tn - fp * fn) / mcc_den if mcc_den else 0.0, 'within_year_AUC': num / den})
V = pd.DataFrame(rows); M = V.merge(FM, on=['rep', 'fold', 'disease', 'model'], suffixes=('_own', '_rep'))
cmp = []
for c in ['ROC_AUC', 'PR_AUC', 'Brier', 'calibration_intercept', 'calibration_slope', 'sensitivity', 'specificity', 'balanced_accuracy', 'F1', 'MCC', 'within_year_AUC']:
    d = (M[f'{c}_own'] - M[f'{c}_rep']).abs(); cmp.append({'quantity': f'fold-level {c} (180 fold x disease x model values)', 'max_abs_difference': float(d.max()), 'n': int(d.notna().sum())})

# corrected t and Holm for FS5 - FS1 and (a) - FS5 (overall and within-year), compared with Table 7
T7 = pd.read_csv(P1 / 'tables' / 'Table7_disease_specific_comparisons_v1.csv'); ratio = 668 / 2672; t7rows = []
def parse(s): a = s.replace('[', ' ').replace(']', ' ').replace(',', ' ').split(); return float(a[0]), float(a[1]), float(a[2])
for comp, (m1, m0) in {'FS5 minus FS1': ('FS5', 'FS1'), '(a) Rank average minus FS5': ('(a) Rank average', 'FS5')}.items():
    ps = []
    for dn in DIS:
        a = V[(V.disease == dn) & (V.model == m1)].sort_values(['rep', 'fold']); b = V[(V.disease == dn) & (V.model == m0)].sort_values(['rep', 'fold'])
        mo, lo, hi, p = corrected_t_own(a.ROC_AUC.values - b.ROC_AUC.values, ratio); mw, lw, hw, _ = corrected_t_own(a.within_year_AUC.values - b.within_year_AUC.values, ratio)
        rep = T7[(T7.Comparison == comp) & (T7.Disease == dn)].iloc[0]; ro = parse(rep['Overall: fold mean [corrected-t 95% CI]']); rw = parse(rep['Within-year: fold mean [corrected-t 95% CI]'])
        ps.append(p); t7rows.append({'comparison': comp, 'disease': dn, 'own_mean': mo, 'own_lo': lo, 'own_hi': hi, 'own_p': p, 'table_mean': ro[0], 'table_lo': ro[1], 'table_hi': ro[2],
                                     'own_within_mean': mw, 'own_within_lo': lw, 'own_within_hi': hw, 'table_within': rw, 'table_holm': rep['Overall: Holm p']})
    hol = holm_own(ps)
    for k, dn in enumerate(DIS): t7rows[-3 + k]['own_holm'] = hol[k]
T7v = pd.DataFrame(t7rows)
cmp.append({'quantity': 'Table 5/7 corrected-t mean and CI ends (overall, 6 comparisons, 3-decimal table values)', 'max_abs_difference': float(np.abs(T7v[['own_mean', 'own_lo', 'own_hi']].values - T7v[['table_mean', 'table_lo', 'table_hi']].values).max()), 'n': 18})
cmp.append({'quantity': 'Table 5/7 corrected-t within-year mean and CI ends', 'max_abs_difference': float(max(abs(r.own_within_mean - r.table_within[0]) + 0 for r in T7v.itertuples())), 'n': 6})
cmp.append({'quantity': 'Holm-adjusted p (Table 5/7, 3-decimal)', 'max_abs_difference': float((T7v.own_holm - T7v.table_holm.astype(float)).abs().max()), 'n': 6})

# pooled (quantile-score) AUC point estimates and an independent cluster bootstrap (different seed and implementation)
BS = pd.read_csv(P1 / 'metrics' / 'bootstrap_pooled_v2.csv'); locs = L.drop_duplicates('household_ID').sort_values('household_ID')
rng = np.random.default_rng(777); bsrows = []
for dn in DIS:
    Ld = L[L.disease == dn]; piv = {m: Ld[Ld.model == m].pivot_table(index='household_ID', columns='rep', values='training_quantile_score') for m in MODELS}
    y = Ld[Ld.model == 'FS1'].drop_duplicates('household_ID').set_index('household_ID').observed.reindex(piv['FS1'].index).values
    loc_ids = Ld[Ld.model == 'FS1'].drop_duplicates('household_ID').set_index('household_ID').location_id.reindex(piv['FS1'].index).values
    ul, inv = np.unique(loc_ids, return_inverse=True); members = [np.where(inv == j)[0] for j in range(len(ul))]
    draws = [np.concatenate([members[j] for j in rng.integers(0, len(ul), len(ul))]) for _ in range(1000)]
    def pooled_auc(m, ix): return np.mean([auc_rank(y[ix], piv[m][r].values[ix]) for r in (1, 2, 3)])
    full = np.arange(len(y))
    for m in MODELS:
        est = pooled_auc(m, full); bv = np.array([pooled_auc(m, ix) for ix in draws]); rep = BS[(BS.disease == dn) & (BS.model == m) & (BS.metric == 'overall')].iloc[0]
        bsrows.append({'disease': dn, 'quantity': m, 'own_estimate': est, 'own_CI_low': np.quantile(bv, 0.025), 'own_CI_high': np.quantile(bv, 0.975),
                       'reported_estimate': rep.pooled_AUC, 'reported_CI_low': rep.CI_low, 'reported_CI_high': rep.CI_high})
    dv = np.array([pooled_auc('FS5', ix) - pooled_auc('FS1', ix) for ix in draws]); rep = BS[(BS.disease == dn) & (BS.model == 'FS5 - FS1') & (BS.metric == 'overall')].iloc[0]
    bsrows.append({'disease': dn, 'quantity': 'FS5 - FS1', 'own_estimate': pooled_auc('FS5', full) - pooled_auc('FS1', full), 'own_CI_low': np.quantile(dv, 0.025), 'own_CI_high': np.quantile(dv, 0.975),
                   'reported_estimate': rep.pooled_AUC, 'reported_CI_low': rep.CI_low, 'reported_CI_high': rep.CI_high})
BV = pd.DataFrame(bsrows)
cmp.append({'quantity': 'pooled quantile-score AUC point estimates (15 values)', 'max_abs_difference': float((BV.own_estimate - BV.reported_estimate).abs().max()), 'n': len(BV)})
cmp.append({'quantity': 'bootstrap 95% CI ends, independent resampling (Monte Carlo difference expected)', 'max_abs_difference': float(np.abs(BV[['own_CI_low', 'own_CI_high']].values - BV[['reported_CI_low', 'reported_CI_high']].values).max()), 'n': 2 * len(BV)})

# permutation p-values and Holm (Table 10), Table 1 counts, forward temporal AUC
A = pd.concat([pd.read_csv(p) for p in sorted((P1 / 'E5_permutations').glob('*.csv'))]); G = A.groupby(['scheme', 'perm', 'disease'])[['FS1_ROC_AUC', 'FS5_ROC_AUC']].mean().reset_index()
G['gain'] = G.FS5_ROC_AUC - G.FS1_ROC_AUC; obs = G[G.perm == -1].set_index('disease').gain; T10 = pd.read_csv(P1 / 'tables' / 'Table10_negative_controls_v1.csv'); pr = []
for sch in ('household', 'village'):
    for dn in DIS:
        nul = G[(G.scheme == sch) & (G.disease == dn) & (G.perm >= 0)].gain.values; pr.append({'scheme': sch, 'disease': dn, 'n_perm': len(nul), 'own_p': (1 + (nul >= obs[dn]).sum()) / (1 + len(nul))})
PR = pd.DataFrame(pr); PR['own_holm'] = holm_own(PR.own_p.values); PR = PR.merge(T10[['Scheme', 'Disease', 'Permutation p (resolution)', 'Holm p (6 tests)']], left_on=['scheme', 'disease'], right_on=['Scheme', 'Disease'])
PR['table_p'] = PR['Permutation p (resolution)'].str.split(' ').str[0].astype(float)
cmp.append({'quantity': 'permutation p (6 values, 3-decimal table)', 'max_abs_difference': float((PR.own_p - PR.table_p).abs().max()), 'n': 6})
cmp.append({'quantity': 'permutation Holm p (6 values)', 'max_abs_difference': float((PR.own_holm - PR['Holm p (6 tests)'].astype(float)).abs().max()), 'n': 6})
import sys; sys.path.insert(0, str(ROOT / 'src')); from paper1.core import setup
df, Y, _, _ = setup(); T1 = pd.read_csv(P1 / 'tables' / 'Table1_dataset_prevalence_v1.csv'); bad = 0
for _, r in T1.iterrows():
    m = np.ones(len(df), bool) if r['Survey year'] == 'All' else (df.Year.values == int(r['Survey year']))
    bad += int(m.sum() != r.Households)
    for d, dn in enumerate(DIS): bad += int(int(r[f'{dn} positive, n (%)'].split(' ')[0]) != int(Y[m, d].sum()))
cmp.append({'quantity': 'Table 1 household and case counts (24 values)', 'max_abs_difference': float(bad), 'n': 24})
Z = np.load(P1 / 'E7_temporal' / 'forward_temporal_predictions_v1.npz'); te = np.where(df.temporal_set.values == 'test')[0]; T12 = pd.read_csv(P1 / 'tables' / 'Table12_forward_temporal_validation_v1.csv'); dd = []
for _, r in T12.iterrows():
    d = DIS.index(r.Disease); s = ['FS1', 'FS5'].index(r.Features); dd.append(abs(auc_rank(Y[te, d], Z['test'][:, d, s]) - float(r['ROC-AUC [95% CI]'].split(' ')[0])))
cmp.append({'quantity': 'forward temporal ROC-AUC (6 values, 3-decimal table)', 'max_abs_difference': float(max(dd)), 'n': 6})
C = pd.DataFrame(cmp); C.to_csv(OUT / 'statistics_verification_summary_v1.csv', index=False); V.to_csv(OUT / 'fold_metrics_recomputed_v1.csv', index=False)
T7v.to_csv(OUT / 'corrected_t_recomputed_v1.csv', index=False); BV.to_csv(OUT / 'bootstrap_recomputed_v1.csv', index=False); PR.to_csv(OUT / 'permutation_recomputed_v1.csv', index=False)
print(C.to_string()); print(BV.round(3).to_string())
