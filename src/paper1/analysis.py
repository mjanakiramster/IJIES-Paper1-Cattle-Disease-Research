"""Paper 1 tables and figures, computed only from saved experiment outputs (results/paper1_v1, Stage 5/9 results and the
audit). No number is typed in by hand. Called by the notebook; can also be run directly:  python -m paper1.analysis"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import (setup, dev_splits, OUT, DIS, auc, pr_auc, within_year_auc, ecdf_rank, platt, youden_threshold, threshold_metrics,
                   calibration, corrected_t, cluster_bootstrap, holm, FEATURE_SETS)
from . import experiments as X

ROOT = Path(__file__).resolve().parents[2]
TAB = OUT / 'tables'; FIG = OUT / 'figures'; TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
COL = {'FS1': '#7a7a74', 'FS5': '#0072B2', '(a) Rank average': '#D55E00', '(b) Inner-selected single': '#009E73'}
YEARS = [2012, 2014, 2015, 2018, 2019]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 9.5,
                     'axes.titleweight': 'bold', 'legend.frameon': False, 'savefig.dpi': 300})
LIB_NAMES = None


def save_fig(fig, name):
    fig.savefig(FIG / f'{name}.png', bbox_inches='tight'); fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight'); plt.close(fig)


def save_tab(df, name):
    df.to_csv(TAB / f'{name}.csv', index=False); return df


# ================================================================ predictions per split (models: FS1, FS5, (a), (b))
def collect_predictions():
    """Per development split: calibrated test predictions, thresholds and inner metrics for the four headline models."""
    cache = OUT / 'predictions' / 'dev_predictions_v2.pkl'
    if cache.exists(): return pd.read_pickle(cache)
    df, Y, _, _ = setup(); rows = []
    for i in range(5, 20):
        z3 = np.load(X.E3 / f'split_{i:02d}.npz'); r, f = int(z3['rep']), int(z3['fold']); z2 = np.load(X.E2 / f'r{r}_f{f}.npz')
        tr, te = z3['train_idx'], z3['test_idx']; assert (tr == z2['train_idx']).all() and (te == z2['test_idx']).all()
        for d, dn in enumerate(DIS):
            ytr, yte = Y[tr, d], Y[te, d]
            # (a) equal-weight ECDF rank average of 23 models; ECDF from training OOF scores
            Po, Pt = z3['oof'][:, d, :], z3['test'][:, d, :]
            U_oof = np.column_stack([ecdf_rank(Po[:, m], Po[:, m]) for m in range(23)]).mean(1)
            U_te = np.column_stack([ecdf_rank(Po[:, m], Pt[:, m]) for m in range(23)]).mean(1)
            inner_auc = np.array([auc(ytr, Po[:, m]) for m in range(23)]); jb = int(np.argmax(inner_auc))
            models = {'FS1': (z2['oof'][:, d, 0], z2['test'][:, d, 0]), 'FS5': (z2['oof'][:, d, 1], z2['test'][:, d, 1]),
                      '(a) Rank average': (U_oof, U_te), '(b) Inner-selected single': (Po[:, jb], Pt[:, jb])}
            for name, (so, st) in models.items():
                p_te = platt(so, ytr, st); p_oof = platt(so, ytr, so); thr = youden_threshold(ytr, p_oof)
                rows.append({'split': i, 'rep': r, 'fold': f, 'disease': dn, 'model': name, 'test_idx': te, 'score': st, 'prob': p_te, 'threshold': thr,
                             'qscore': ecdf_rank(so, st), 'platt_fallback': bool(np.all(p_te == p_te[0])),
                             'selected_single': jb if name.startswith('(b)') else None})
    P = pd.DataFrame(rows); cache.parent.mkdir(parents=True, exist_ok=True); P.to_pickle(cache); return P


def fold_metrics(P):
    df, Y, _, _ = setup(); yr = df.Year.values; rows = []
    for _, r in P.iterrows():
        d = DIS.index(r.disease); y = Y[r.test_idx, d]; s = r.score; p = r.prob
        wy, wm, per = within_year_auc(y, s, yr[r.test_idx])
        rec = {'split': r.split, 'rep': r.rep, 'fold': r.fold, 'disease': r.disease, 'model': r.model, 'n_test': len(y), 'positives': int(y.sum()),
               'ROC_AUC': auc(y, s), 'PR_AUC': pr_auc(y, s), 'prevalence': y.mean(), 'within_year_AUC': wy, 'within_year_AUC_macro': wm, 'platt_fallback': r.platt_fallback}
        rec.update(calibration(y, p)); rec.update(threshold_metrics(y, p, r.threshold)); rows.append(rec)
    return pd.DataFrame(rows)


def pooled(P, model, disease, rep, kind='qscore'):
    """Pooled out-of-fold predictions for one repeat (every household once). kind='qscore': each test score expressed as its
    quantile in the training out-of-fold score distribution (monotone within a fold, comparable across folds) - used for
    pooled discrimination; kind='prob': Platt-calibrated probabilities - used for calibration plots."""
    q = P[(P.model == model) & (P.disease == disease) & (P.rep == rep)]
    idx = np.concatenate(q.test_idx.values); v = np.concatenate(q[kind].values)
    o = np.argsort(idx); return idx[o], v[o]


def bootstrap_table(P, B=1000):
    """Location-cluster bootstrap CIs for pooled (per-repeat, then averaged) overall and within-year AUC, and FS5 - FS1 differences."""
    cache = OUT / 'metrics' / 'bootstrap_v2.pkl'
    if cache.exists(): return pd.read_pickle(cache)
    df, Y, _, _ = setup(); loc = df['loc'].values; yr = df.Year.values; rows = []
    models = list(COL)
    for dn in DIS:
        d = DIS.index(dn); pr = {(m, r): pooled(P, m, dn, r)[1] for m in models for r in (1, 2, 3)}; y = Y[:, d]
        def stat(ix, m, kind):
            vals = [auc(y[ix], pr[(m, r)][ix]) if kind == 'overall' else within_year_auc(y[ix], pr[(m, r)][ix], yr[ix])[0] for r in (1, 2, 3)]
            return float(np.mean(vals))
        full = np.arange(len(y))
        for kind in ('overall', 'within_year'):
            for m in models:
                lo, hi, _ = cluster_bootstrap(loc, lambda ix: stat(ix, m, kind), B=B)
                rows.append({'disease': dn, 'model': m, 'metric': kind, 'pooled_AUC': stat(full, m, kind), 'CI_low': lo, 'CI_high': hi})
            for a, b in (('FS5', 'FS1'), ('(a) Rank average', 'FS5'), ('(a) Rank average', 'FS1')):
                lo, hi, vals = cluster_bootstrap(loc, lambda ix: stat(ix, a, kind) - stat(ix, b, kind), B=B)
                rows.append({'disease': dn, 'model': f'{a} - {b}', 'metric': kind, 'pooled_AUC': stat(full, a, kind) - stat(full, b, kind), 'CI_low': lo, 'CI_high': hi,
                             'p_two_sided': float(min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean())))})
    R = pd.DataFrame(rows); cache.parent.mkdir(parents=True, exist_ok=True); R.to_pickle(cache); return R


# ================================================================ tables
def table1():
    df, Y, _, _ = setup(); rows = []
    for v in YEARS + ['All']:
        m = np.ones(len(df), bool) if v == 'All' else df.Year.values == v
        rows.append({'Survey year': v, 'Households': int(m.sum()), 'Locations': int(df.loc[m, 'loc'].nunique()),
                     **{f'{d} positive, n (%)': f'{int(Y[m, k].sum())} ({100 * Y[m, k].mean():.1f})' for k, d in enumerate(DIS)}})
    return save_tab(pd.DataFrame(rows), 'Table1_dataset_prevalence_v1')


def table2():
    rows = [('FS1', 'Environmental', 'Altitude, temperature, precipitation, relative humidity, wind speed', 5, 'No'),
            ('FS5', 'Environmental (as FS1)', 'As FS1', 5, 'No'),
            ('FS5', 'Climate indices', 'Sea-level temperature, De Martonne index, temperature x RH, log precipitation x RH, log precipitation, wind x RH', 6, 'No'),
            ('FS5', 'Coordinates', 'Latitude, longitude, latitude + longitude, latitude - longitude', 4, 'No'),
            ('FS5', 'Kernel disease rates', 'Gaussian-kernel rates at 50, 150, 400 km for CBPP, FMD and LSD (own and cross-disease)', 9, 'Yes (training rows, own 81 km block excluded)'),
            ('FS5', 'k-nearest-location rates', 'Rates over the 5, 15, 40 nearest training locations for each disease', 9, 'Yes (training rows, own block excluded)'),
            ('FS5', 'Sampling density', 'log kernel density of training households (3) and log distance to the k-th nearest training location (3)', 6, 'No (household counts only)')]
    return save_tab(pd.DataFrame(rows, columns=['Feature set', 'Group', 'Variables', 'Number', 'Uses outcome labels?']), 'Table2_feature_sets_v1')


def table3():
    rows = [('ExtraTrees (FS1, FS5)', 'Feature-set comparison', '300 trees, min_samples_leaf 10, max_features sqrt'),
            ('GAM (B, EL, E)', '23-model library', 'Cubic splines (5 knots) + logistic regression, C = 0.5'),
            ('Random forest (B, EL)', '23-model library', '200 trees, min_samples_leaf 10, max_features sqrt'),
            ('XGBoost (B, EL)', '23-model library', '300 trees, depth 4, learning rate 0.05, subsample 0.8, colsample 0.8, min_child_weight 5'),
            ('CatBoost (B, E, EL) and variants', '23-model library', '300 iterations, depth 5, learning rate 0.05; group-balanced, village-level and multi-task variants'),
            ('k-nearest neighbours (E)', '23-model library', 'k = 50, standardised inputs'),
            ('Logistic regression (EL, ELK)', '23-model library', 'L2, C = 0.5, standardised inputs'),
            ('ExtraTrees (EL, ELK)', '23-model library', '200 trees, min_samples_leaf 10, max_features sqrt'),
            ('MLP (EL)', '23-model library', 'Hidden layers (32, 16), alpha 0.01, 400 iterations'),
            ('GP kriging (G)', '23-model library', 'Location-level Gaussian process on empirical logits'),
            ('LightGBM (ELK)', '23-model library', '300 trees, 15 leaves, learning rate 0.05, subsample 0.8, colsample 0.8'),
            ('SVM-RBF (ELK)', '23-model library', 'C = 1, gamma = scale, standardised inputs; Platt scaling on inner OOF'),
            ('(a) Rank average', 'Final ensemble', 'Equal-weight mean of ECDF ranks of the 23 models; Platt calibration and Youden threshold on inner OOF'),
            ('(b) Inner-selected single', 'Comparator', 'Library model with the best mean inner-fold ROC-AUC in each outer split'),
            ('SGO, MSGO, PSO, GA, DE, GWO, random search', 'Optimisation study (Stages 5-6)', '10 agents, 150 fitness evaluations per run, 45 runs per optimiser (3 diseases x 5 folds x 3 seeds); fitness = mean - 0.5 SD inner AUC - 0.005 complexity')]
    return save_tab(pd.DataFrame(rows, columns=['Model', 'Role', 'Fixed hyperparameters']), 'Table3_models_v2')


def table4_5(FM, BS):
    rows = []
    for dn in DIS:
        for m in COL:
            g = FM[(FM.disease == dn) & (FM.model == m)]; b = BS[(BS.disease == dn) & (BS.model == m)]
            bo = b[b.metric == 'overall'].iloc[0]; bw = b[b.metric == 'within_year'].iloc[0]
            rows.append({'Disease': dn, 'Model': m, 'ROC-AUC fold mean (SD)': f'{g.ROC_AUC.mean():.3f} ({g.ROC_AUC.std():.3f})',
                         'ROC-AUC pooled [95% CI]': f'{bo.pooled_AUC:.3f} [{bo.CI_low:.3f}, {bo.CI_high:.3f}]',
                         'PR-AUC (no-skill)': f'{g.PR_AUC.mean():.3f} ({g.prevalence.mean():.3f})', 'Brier': f'{g.Brier.mean():.3f}',
                         'Cal. intercept (median)': f'{g.calibration_intercept.median():.2f}', 'Cal. slope (median)': f'{g.calibration_slope.median():.2f}', 'Platt fallback folds': f'{int(g.platt_fallback.sum())}/15',
                         'Sensitivity': f'{g.sensitivity.mean():.2f}', 'Specificity': f'{g.specificity.mean():.2f}', 'Balanced acc.': f'{g.balanced_accuracy.mean():.2f}',
                         'F1': f'{g.F1.mean():.2f}', 'MCC': f'{g.MCC.mean():.2f}',
                         'Within-year AUC pooled [95% CI]': f'{bw.pooled_AUC:.3f} [{bw.CI_low:.3f}, {bw.CI_high:.3f}]',
                         'Within-year AUC fold mean (macro)': f'{g.within_year_AUC.mean():.3f} ({g.within_year_AUC_macro.mean():.3f})'})
    T = pd.DataFrame(rows)
    save_tab(T[['Disease', 'Model', 'ROC-AUC fold mean (SD)', 'ROC-AUC pooled [95% CI]', 'PR-AUC (no-skill)', 'Brier', 'Cal. intercept (median)', 'Cal. slope (median)', 'Platt fallback folds',
                'Sensitivity', 'Specificity', 'Balanced acc.', 'F1', 'MCC']], 'Table4_overall_spatial_performance_v1')
    return save_tab(T[['Disease', 'Model', 'ROC-AUC pooled [95% CI]', 'Within-year AUC pooled [95% CI]', 'Within-year AUC fold mean (macro)']], 'Table5_within_year_performance_v1')


def per_year_auc(P):
    df, Y, _, _ = setup(); yr = df.Year.values; rows = []
    for dn in DIS:
        d = DIS.index(dn)
        for m in COL:
            for r in (1, 2, 3):
                idx, p = pooled(P, m, dn, r); y = Y[idx, d]
                for v in YEARS:
                    k = yr[idx] == v; rows.append({'disease': dn, 'model': m, 'rep': r, 'year': v, 'n': int(k.sum()), 'positives': int(y[k].sum()), 'AUC': auc(y[k], p[k])})
    R = pd.DataFrame(rows).groupby(['disease', 'model', 'year']).agg(n=('n', 'first'), positives=('positives', 'first'), AUC=('AUC', 'mean')).reset_index()
    return save_tab(R, 'TableS_year_specific_AUC_v1')


def table6_7(FM, BS):
    conf = json.load(open(ROOT / 'configs/decision_log.json'))['confirmatory_results'] if 'confirmatory_results' in json.load(open(ROOT / 'configs/decision_log.json')) else {}
    S9 = pd.read_csv(ROOT / 'results/stage9_confirmatory_v1/confirmatory_summary.csv') if (ROOT / 'results/stage9_confirmatory_v1/confirmatory_summary.csv').exists() else None
    perm = pd.read_csv(OUT / 'tables' / 'Table10_negative_controls_v1.csv') if (OUT / 'tables' / 'Table10_negative_controls_v1.csv').exists() else None
    rows = []
    for comp in (('FS5', 'FS1'), ('(a) Rank average', 'FS5')):
        p_over, p_wy = [], []
        for dn in DIS:
            a = FM[(FM.disease == dn) & (FM.model == comp[0])].sort_values('split'); b = FM[(FM.disease == dn) & (FM.model == comp[1])].sort_values('split')
            d_over = a.ROC_AUC.values - b.ROC_AUC.values; d_wy = a.within_year_AUC.values - b.within_year_AUC.values
            m1, l1, h1, p1 = corrected_t(d_over, 2672, 668, 15); m2, l2, h2, p2 = corrected_t(d_wy, 2672, 668, 15); p_over.append(p1); p_wy.append(p2)
            bo = BS[(BS.disease == dn) & (BS.model == f'{comp[0]} - {comp[1]}') & (BS.metric == 'overall')].iloc[0]
            bw = BS[(BS.disease == dn) & (BS.model == f'{comp[0]} - {comp[1]}') & (BS.metric == 'within_year')].iloc[0]
            rows.append({'Comparison': f'{comp[0]} minus {comp[1]}', 'Disease': dn, 'Overall: fold mean [corrected-t 95% CI]': f'{m1:+.3f} [{l1:+.3f}, {h1:+.3f}]',
                         'Overall: folds > 0': f'{(d_over > 0).sum()}/15', 'Overall: pooled [bootstrap 95% CI]': f'{bo.pooled_AUC:+.3f} [{bo.CI_low:+.3f}, {bo.CI_high:+.3f}]',
                         'Within-year: fold mean [corrected-t 95% CI]': f'{m2:+.3f} [{l2:+.3f}, {h2:+.3f}]', 'Within-year: folds > 0': f'{(d_wy > 0).sum()}/15',
                         'Within-year: pooled [bootstrap 95% CI]': f'{bw.pooled_AUC:+.3f} [{bw.CI_low:+.3f}, {bw.CI_high:+.3f}]', '_p_over': p1, '_p_wy': p2})
        k = len(rows) - 3
        for j, (po, pw) in enumerate(zip(holm(p_over), holm(p_wy))):
            rows[k + j]['Overall: Holm p'] = f'{po:.3f}'; rows[k + j]['Within-year: Holm p'] = f'{pw:.3f}'
    T = pd.DataFrame(rows).drop(columns=['_p_over', '_p_wy']); save_tab(T, 'Table7_disease_specific_comparisons_v1')
    return save_tab(T[(T.Comparison == 'FS5 minus FS1') & (T.Disease == 'FMD')], 'Table6_FMD_FS5_vs_FS1_v1')


def table8_9():
    A = pd.concat([pd.read_csv(p) for p in sorted(X.E6.glob('*.csv'))], ignore_index=True); A['gain'] = A.FS5_ROC_AUC - A.FS1_ROC_AUC
    A.to_csv(OUT / 'sensitivity' / 'E6_fold_results_v1.csv', index=False) if (OUT / 'sensitivity').mkdir(exist_ok=True) is None else None
    rows = []
    for (c, dn), g in A.groupby(['condition', 'disease']):
        m, lo, hi, p = corrected_t(g.gain.values, g.train_households.mean(), g.test_households.mean(), 15)
        rows.append({'Condition': c, 'Disease': dn, 'Training households': f'{g.train_households.mean():.0f}', 'Training locations': f'{g.train_locations.mean():.0f}',
                     'Removed households': f'{g.removed_households.mean():.0f}', 'Min test positives': int(g.test_positives.min()),
                     'Median nearest training location (km)': f'{g.median_nearest_training_location_km.mean():.0f}',
                     'FS1 ROC-AUC': f'{g.FS1_ROC_AUC.mean():.3f}', 'FS5 ROC-AUC': f'{g.FS5_ROC_AUC.mean():.3f}', 'FS5 - FS1 [95% CI]': f'{m:+.3f} [{lo:+.3f}, {hi:+.3f}]',
                     'Folds > 0': f'{(g.gain > 0).sum()}/15', 'FS1 PR-AUC': f'{g.FS1_PR_AUC.mean():.3f}', 'FS5 PR-AUC': f'{g.FS5_PR_AUC.mean():.3f}', 'No-skill PR-AUC': f'{g.test_prevalence.mean():.3f}'})
    T = pd.DataFrame(rows); order = {c: i for i, c in enumerate(X.E6_CONDITIONS)}; T = T.sort_values(['Disease', 'Condition'], key=lambda s: s.map(order) if s.name == 'Condition' else s)
    save_tab(T[T.Condition.isin(['base81', 'blocks120', 'blocks160'])], 'Table8_block_size_sensitivity_v1')
    return save_tab(T[~T.Condition.isin(['blocks120', 'blocks160'])], 'Table9_proximity_removal_v1'), A


def table10():
    A = pd.concat([pd.read_csv(p) for p in sorted(X.E5.glob('*.csv'))], ignore_index=True)
    M = A.groupby(['scheme', 'perm', 'disease'])[['FS1_ROC_AUC', 'FS5_ROC_AUC', 'FS1_PR_AUC', 'FS5_PR_AUC', 'test_prevalence']].mean().reset_index()
    M['gain'] = M.FS5_ROC_AUC - M.FS1_ROC_AUC; obs = M[M.perm == -1].set_index('disease'); rows = []
    for (s, dn), g in M[M.perm >= 0].groupby(['scheme', 'disease']):
        o = obs.loc[dn]; n = len(g)
        rows.append({'Scheme': s, 'Disease': dn, 'Permutations': n, 'Null FS1 ROC-AUC mean [2.5-97.5%]': f'{g.FS1_ROC_AUC.mean():.3f} [{g.FS1_ROC_AUC.quantile(.025):.3f}, {g.FS1_ROC_AUC.quantile(.975):.3f}]',
                     'Null FS5 ROC-AUC mean [2.5-97.5%]': f'{g.FS5_ROC_AUC.mean():.3f} [{g.FS5_ROC_AUC.quantile(.025):.3f}, {g.FS5_ROC_AUC.quantile(.975):.3f}]',
                     'Null gain mean [2.5-97.5%]': f'{g.gain.mean():+.3f} [{g.gain.quantile(.025):+.3f}, {g.gain.quantile(.975):+.3f}]',
                     'Observed gain': f'{o.gain:+.3f}', 'Permutation p (resolution)': f'{(1 + (g.gain >= o.gain).sum()) / (1 + n):.3f} ({1 / (1 + n):.3f})',
                     'Null FS5 PR-AUC': f'{g.FS5_PR_AUC.mean():.3f}', 'No-skill PR-AUC': f'{g.test_prevalence.mean():.3f}', '_p': (1 + (g.gain >= o.gain).sum()) / (1 + n)})
    T = pd.DataFrame(rows); T['Holm p (6 tests)'] = [f'{v:.3f}' for v in holm(T._p.values)]
    return save_tab(T.drop(columns='_p'), 'Table10_negative_controls_v1'), M


def table11():
    """Ensemble prior/tie sensitivity: Stage 4 cache (original code) vs audit A5 (corrected prior) vs final (prior + ties)."""
    df, Y, _, _ = setup(); C = np.load(ROOT / 'results/stage4_v1/cache/library23_predictions_v1.npz'); rows = []
    for i in range(5, 20):
        z3 = np.load(X.E3 / f'split_{i:02d}.npz'); te = z3['test_idx']; a5 = np.load(X.A5_PARTS / f'split_{i:02d}.npz')
        G_aff = [j for j in range(21) if j in range(6, 16) or j in (18, 19, 20)] + [21, 22]
        for d, dn in enumerate(DIS):
            Pt0, Po0 = C[f'test_{i}'][:, d, :].astype(float), C[f'oof_{i}'][:, d, :].astype(float)
            Pt1, Po1 = Pt0.copy(), Po0.copy(); Pt1[:, G_aff] = a5['test'][:, d, :]; Po1[:, G_aff] = a5['oof'][:, d, :]
            Pt2, Po2 = z3['test'][:, d, :], z3['oof'][:, d, :]
            rec = {'split': i, 'disease': dn}
            for lab, (Pt, Po) in {'original code': (Pt0, Po0), 'corrected prior': (Pt1, Po1), 'corrected prior + deterministic ties': (Pt2, Po2)}.items():
                rec[lab] = auc(Y[te, d], np.column_stack([ecdf_rank(Po[:, m], Pt[:, m]) for m in range(23)]).mean(1))
            rows.append(rec)
    R = pd.DataFrame(rows); out = []
    for dn, g in R.groupby('disease'):
        for lab in ('corrected prior', 'corrected prior + deterministic ties'):
            m, lo, hi, p = corrected_t((g[lab] - g['original code']).values, 2672, 668, 15)
            out.append({'Disease': dn, 'Version': lab, 'Rank-average ROC-AUC (original)': f"{g['original code'].mean():.4f}", 'Rank-average ROC-AUC (version)': f'{g[lab].mean():.4f}',
                        'Difference [corrected-t 95% CI]': f'{m:+.4f} [{lo:+.4f}, {hi:+.4f}]', 'Max fold |difference|': f"{(g[lab] - g['original code']).abs().max():.4f}"})
    return save_tab(pd.DataFrame(out), 'Table11_ensemble_prior_sensitivity_v1')


def table12():
    df, Y, _, _ = setup(); z = np.load(X.E7 / 'forward_temporal_predictions_v1.npz'); te = z['test_idx']; tr = z['train_idx']; loc = df['loc'].values[te]; rows = []
    for d, dn in enumerate(DIS):
        y = Y[te, d]
        for s, name in enumerate(FEATURE_SETS):
            p = z['test'][:, d, s]; lo, hi, _ = cluster_bootstrap(loc, lambda ix: auc(y[ix], p[ix]), B=1000)
            rows.append({'Disease': dn, 'Features': name, 'Train households (2012-2015)': len(tr), 'Test households (2018-2019)': len(te),
                         'Train prevalence': f'{Y[tr, d].mean():.3f}', 'Test prevalence': f'{y.mean():.3f}', 'ROC-AUC [95% CI]': f'{auc(y, p):.3f} [{lo:.3f}, {hi:.3f}]',
                         'PR-AUC (no-skill)': f'{pr_auc(y, p):.3f} ({y.mean():.3f})'})
    return save_tab(pd.DataFrame(rows), 'Table12_forward_temporal_validation_v1')


def supplementary_tables():
    S5 = pd.read_csv(ROOT / 'results/stage5_v1/optimiser_real_summary.csv') if (ROOT / 'results/stage5_v1/optimiser_real_summary.csv').exists() else None
    if S5 is not None: S5.to_csv(TAB / 'TableS_optimisation_stage5_as_run_v1.csv', index=False)
    for f in ('confirmatory_summary.csv', 'confirmatory_fold_metrics.csv'):
        p = ROOT / 'results/stage9_confirmatory_v1' / f
        if p.exists(): pd.read_csv(p).to_csv(TAB / f'TableS_confirmatory_as_observed_{f.replace(".csv", "")}_v1.csv', index=False)
    imp = pd.concat([pd.read_csv(p) for p in sorted(X.E8.glob('*.csv'))], ignore_index=True)
    I = imp.groupby(['disease', 'group']).agg(mean_AUC_drop=('mean_AUC_drop', 'mean'), sd_over_folds=('mean_AUC_drop', 'std'), folds=('mean_AUC_drop', 'size')).reset_index()
    save_tab(I, 'TableS_grouped_permutation_importance_v1'); return I


# ================================================================ figures
def fig_workflow():
    from matplotlib.patches import FancyBboxPatch
    steps = ['ILRI household survey\n3,340 households\n462 locations, 5 years', 'Exclude Weight, Year,\nRegion, identifiers', 'Spatial blocks (81 km)\n3 x 5 development folds',
             'Features built from\ntraining rows only\n(own block excluded)', 'Models: ExtraTrees FS1/FS5,\n23-model rank average', 'Metrics: overall and\nwithin-survey-year AUC,\nPR-AUC, calibration',
             'Sensitivity: permutations,\nproximity, block size,\nforward temporal test']
    fig, ax = plt.subplots(figsize=(7.2, 2.3)); ax.axis('off'); ax.set_xlim(0, 7 * 1.03); ax.set_ylim(0, 1.25)
    for i, s in enumerate(steps):
        x = i * 1.03 + 0.02; ax.add_patch(FancyBboxPatch((x, 0.15), 0.9, 0.95, boxstyle='round,pad=0.015', fc='#eef4fb' if i in (2, 3, 4) else '#f3f3f0', ec='#55554f', lw=0.7))
        ax.text(x + 0.45, 0.625, s, ha='center', va='center', fontsize=6.4)
        if i < 6: ax.annotate('', xy=(x + 1.025, 0.625), xytext=(x + 0.925, 0.625), arrowprops=dict(arrowstyle='->', color='#55554f', lw=0.8))
    save_fig(fig, 'Fig1_workflow_v1')


def fig_years():
    df, Y, _, _ = setup(); fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={'width_ratios': [1.05, 1]})
    cols = dict(zip(YEARS, ['#E69F00', '#56B4E9', '#009E73', '#0072B2', '#CC79A7']))
    L = df.groupby('loc').agg(lat=('Latitude', 'first'), lon=('Longitude', 'first'), year=('Year', lambda s: s.mode().iloc[0]))
    for v in YEARS:
        q = L[L.year == v]; axs[0].scatter(q.lon, q.lat, s=7, color=cols[v], label=f'{v} (n={len(q)})', alpha=0.85, edgecolor='white', linewidth=0.2)
    axs[0].set_xlabel('Longitude (degrees E)'); axs[0].set_ylabel('Latitude (degrees N)'); axs[0].set_title('(a) Survey locations by survey year', loc='left')
    axs[0].legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, markerscale=1.5); axs[0].set_aspect('equal')
    w = 0.26; x = np.arange(len(YEARS))
    for k, (dn, c) in enumerate(zip(DIS, ['#7a7a74', '#0072B2', '#D55E00'])):
        prev = [Y[df.Year.values == v, k].mean() * 100 for v in YEARS]; axs[1].bar(x + (k - 1) * w, prev, w, color=c, label=dn)
    axs[1].set_xticks(x); axs[1].set_xticklabels(YEARS); axs[1].set_ylabel('Households reporting the disease (%)'); axs[1].set_title('(b) Reported prevalence by survey year', loc='left'); axs[1].legend(fontsize=7)
    fig.tight_layout(); save_fig(fig, 'Fig2_survey_years_v2')


def fig_overall_vs_within(BS):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True); models = list(COL)
    for ax, dn in zip(axs, DIS):
        for j, m in enumerate(models):
            for k, (metric, mk) in enumerate((('overall', 'o'), ('within_year', 's'))):
                b = BS[(BS.disease == dn) & (BS.model == m) & (BS.metric == metric)].iloc[0]; xx = j + (k - 0.5) * 0.3
                ax.errorbar(xx, b.pooled_AUC, yerr=[[b.pooled_AUC - b.CI_low], [b.CI_high - b.pooled_AUC]], fmt=mk, color=COL[m], mfc=COL[m] if k == 0 else 'white', ms=5, capsize=2, lw=1)
        ax.axhline(0.5, color='#999', lw=0.8, ls=':'); ax.set_xticks(range(len(models))); ax.set_xticklabels(['FS1', 'FS5', '(a)', '(b)']); ax.set_title(dn, loc='left')
    axs[0].set_ylabel('Pooled ROC-AUC (95% cluster-bootstrap CI)')
    from matplotlib.lines import Line2D
    axs[2].legend(handles=[Line2D([], [], marker='o', color='#444', ls='', label='Overall (pooled)'), Line2D([], [], marker='s', mfc='white', color='#444', ls='', label='Within survey year')], fontsize=7, loc='upper right')
    fig.tight_layout(); save_fig(fig, 'Fig3_overall_vs_within_year_AUC_v1')


def fig_roc_pr(P):
    from sklearn.metrics import roc_curve, precision_recall_curve
    df, Y, _, _ = setup(); fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6))
    for c, dn in enumerate(DIS):
        d = DIS.index(dn)
        for m in COL:
            idx, p = pooled(P, m, dn, 1, 'qscore'); y = Y[idx, d]; fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
            axs[0, c].plot(fpr, tpr, color=COL[m], lw=1.1, label=f'{m.split(" ")[0]} {auc(y, p):.3f}'); axs[1, c].plot(rc, pr, color=COL[m], lw=1.1, label=f'{m.split(" ")[0]} {pr_auc(y, p):.3f}')
        axs[0, c].plot([0, 1], [0, 1], ':', color='#999', lw=0.8); axs[1, c].axhline(Y[:, d].mean(), ls=':', color='#999', lw=0.8)
        axs[0, c].set_title(f'{dn}: ROC (repeat 1, pooled)', loc='left'); axs[1, c].set_title(f'{dn}: precision-recall', loc='left')
        axs[0, c].set_xlabel('False-positive rate'); axs[1, c].set_xlabel('Recall'); axs[0, c].legend(fontsize=6, title='AUC', title_fontsize=6); axs[1, c].legend(fontsize=6, title='PR-AUC', title_fontsize=6)
    axs[0, 0].set_ylabel('True-positive rate'); axs[1, 0].set_ylabel('Precision'); fig.tight_layout(); save_fig(fig, 'FigS1_ROC_PR_curves_v1')


def fig_calibration(P):
    df, Y, _, _ = setup(); fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, dn in zip(axs, DIS):
        d = DIS.index(dn)
        for m in ('FS5', '(a) Rank average'):
            idx, p = pooled(P, m, dn, 1, 'prob'); y = Y[idx, d]; bins = np.unique(np.quantile(p, np.linspace(0, 1, 11))); k = np.clip(np.digitize(p, bins[1:-1]), 0, len(bins) - 2)
            ax.plot([p[k == j].mean() for j in range(len(bins) - 1)], [y[k == j].mean() for j in range(len(bins) - 1)], 'o-', color=COL[m], ms=3, lw=1, label=m.split(' ')[0])
        ax.plot([0, 1], [0, 1], ':', color='#999', lw=0.8); lim = 0.75; ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_title(dn, loc='left'); ax.set_xlabel('Mean predicted probability (decile)')
    axs[0].set_ylabel('Observed proportion'); axs[0].legend(fontsize=7); fig.tight_layout(); save_fig(fig, 'FigS2_calibration_v1')


def fig_confusion(P):
    df, Y, _, _ = setup(); fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, dn in zip(axs, DIS):
        q = P[(P.model == '(a) Rank average') & (P.disease == dn) & (P.rep == 1)]; d = DIS.index(dn); M = np.zeros((2, 2), int)
        for _, r in q.iterrows():
            y = Y[r.test_idx, d]; yh = (r.prob >= r.threshold).astype(int)
            for a in (0, 1):
                for b in (0, 1): M[a, b] += int(((y == a) & (yh == b)).sum())
        ax.imshow(M, cmap='Blues'); ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(['Pred. 0', 'Pred. 1']); ax.set_yticklabels(['Obs. 0', 'Obs. 1'])
        for a in (0, 1):
            for b in (0, 1): ax.text(b, a, M[a, b], ha='center', va='center', color='white' if M[a, b] > M.max() / 2 else '#222', fontsize=8)
        ax.set_title(f'{dn}: (a), repeat 1', loc='left')
    fig.tight_layout(); save_fig(fig, 'FigS3_confusion_matrices_v1')


def fig_sensitivity(A6):
    conds = X.E6_CONDITIONS
    lab = ['81 km blocks (base)', 'Buffer 25 km', 'Size-matched 25 km', 'Buffer 50 km', 'Size-matched 50 km', 'Buffer 100 km', 'Size-matched 100 km', '120 km blocks', '160 km blocks']
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True); yy = np.arange(len(conds))[::-1]
    for ax, dn in zip(axs, DIS):
        vals, los, his, cs = [], [], [], []
        for c in conds:
            g = A6[(A6.condition == c) & (A6.disease == dn)]; m, lo, hi, _ = corrected_t(g.gain.values, g.train_households.mean(), g.test_households.mean(), 15)
            vals.append(m); los.append(m - lo); his.append(hi - m); cs.append('#0d3f86' if c == 'base81' else '#0072B2' if c.startswith('buffer') else '#a9a8a0' if c.startswith('matched') else '#009E73')
        ax.barh(yy, vals, xerr=[los, his], color=cs, height=0.65, capsize=2, error_kw={'lw': 0.8}); ax.axvline(0, color='#555', lw=0.7)
        ax.set_title(dn, loc='left'); ax.set_xlabel('FS5 - FS1 ROC-AUC')
    axs[0].set_yticks(yy); axs[0].set_yticklabels(lab, fontsize=7)
    fig.text(0.99, 0.005, 'Bars: mean over 15 development folds; whiskers: corrected resampled t 95% CI', ha='right', fontsize=6.5, color='#555')
    fig.tight_layout(); save_fig(fig, 'Fig4_spatial_sensitivity_v2')


def fig_permutation(M):
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 3.9), sharex='col')
    obs = M[M.perm == -1].set_index('disease')
    for i, s in enumerate(('household', 'village')):
        for j, dn in enumerate(DIS):
            ax = axs[i, j]; g = M[(M.scheme == s) & (M.disease == dn) & (M.perm >= 0)]
            ax.hist(g.gain, bins=20, color='#a9a8a0'); ax.axvline(obs.loc[dn, 'gain'], color='#D55E00', lw=1.6); ax.axvline(0, color='#555', lw=0.6, ls=':')
            ax.set_title(f'{dn}, {s}-level null (n={len(g)})', loc='left', fontsize=8)
            if i == 1: ax.set_xlabel('FS5 - FS1 ROC-AUC')
            if j == 0: ax.set_ylabel('Permutations')
    fig.tight_layout(); save_fig(fig, 'Fig5_permutation_null_v1')


def fig_importance(I):
    groups = ['Own-disease rates (6)', 'Cross-disease rates (12)', 'Sampling density (6)', 'Coordinates (4)', 'Climate indices (6)', 'Environmental (5)']
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, dn in zip(axs, DIS):
        g = I[I.disease == dn].set_index('group').reindex(groups)
        ax.barh(range(len(groups)), g.mean_AUC_drop, xerr=g.sd_over_folds, color='#0072B2', height=0.6, error_kw={'lw': 0.7}); ax.axvline(0, color='#555', lw=0.6)
        ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups, fontsize=7); ax.set_title(dn, loc='left'); ax.set_xlabel('ROC-AUC drop when permuted')
    fig.tight_layout(); save_fig(fig, 'FigS4_grouped_importance_v1')


def run_all():
    P = collect_predictions(); FM = fold_metrics(P); (OUT / 'metrics').mkdir(exist_ok=True); FM.to_csv(OUT / 'metrics' / 'fold_metrics_v2.csv', index=False)
    BS = bootstrap_table(P); BS.to_csv(OUT / 'metrics' / 'bootstrap_pooled_v2.csv', index=False)
    t = {'T1': table1(), 'T2': table2(), 'T3': table3()}; t['T5'] = table4_5(FM, BS); t['TS_year'] = per_year_auc(P)
    t['T10'], M = table10(); t['T6'] = table6_7(FM, BS); t['T9'], A6 = table8_9(); t['T11'] = table11(); t['T12'] = table12(); I = supplementary_tables()
    fig_workflow(); fig_years(); fig_overall_vs_within(BS); fig_roc_pr(P); fig_calibration(P); fig_confusion(P); fig_sensitivity(A6); fig_permutation(M); fig_importance(I)
    return FM, BS


if __name__ == '__main__':
    run_all()
