"""Revision-2 analysis: tables and figures for the approved items A-G (results/paper1_rev2_v1/tables, figures).
A ablation FS1-FS5; B SHAP and partial dependence; C1 area of applicability at surveyed locations; D confidence intervals
for PR-AUC, Brier, calibration and threshold metrics, with calibration bands; E campaign identification; F ensemble without
the group-balanced model; G train-test distance distributions; E10b check that saved campaign predictions reproduce Table 6.
Run from the project root: python src/paper1/rev2_analysis_v1.py"""
import sys, json, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from paper1.core import setup, auc, pr_auc, ecdf_rank, within_year_auc, corrected_t, holm, DIS
from paper1.rev2_experiments_v1 import ABLATION
ROOT = Path(__file__).resolve().parents[2]; P1 = ROOT / 'results/paper1_v1'; REV = ROOT / 'results/paper1_rev_v1'; R2 = ROOT / 'results/paper1_rev2_v1'
TAB, FIG = R2 / 'tables', R2 / 'figures'; TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold', 'legend.frameon': False, 'savefig.dpi': 300})
df, Y, NB, FB = setup(); LOC = df['loc'].values; YR = df.Year.values; RATIO = 668 / 2672; B = 1000; SEEDB = 20261122
SETS = ['FS1', 'FS2', 'FS3', 'FS4', 'FS5']; NFEAT = {'FS1': 5, 'FS2': 11, 'FS3': 15, 'FS4': 33, 'FS5': 39}
SET_COL = {'FS1': '#b9b9b2', 'FS2': '#9ecae1', 'FS3': '#4292c6', 'FS4': '#08519c', 'FS5': '#D55E00'}
GROUP_COL = {'Environmental': '#7a7a74', 'Climate indices': '#56B4E9', 'Coordinates': '#009E73', 'Own-disease rates': '#0072B2', 'Cross-disease rates': '#D55E00', 'Sampling density': '#CC79A7'}
def save(fig, name): fig.savefig(FIG / f'{name}.png', bbox_inches='tight'); fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight'); plt.close(fig)
def splits(): return [(r, f) for r in (1, 2, 3) for f in range(5)]
uloc = np.unique(LOC); MEMBERS = [np.where(LOC == l)[0] for l in uloc]
DRAWS = None
def draws():
    global DRAWS
    if DRAWS is None:
        rng = np.random.default_rng(SEEDB); DRAWS = [np.concatenate([MEMBERS[j] for j in rng.integers(0, len(uloc), len(uloc))]) for _ in range(B)]
    return DRAWS
def ci_str(m, lo, hi, sign=False): return (f'{m:+.3f} [{lo:+.3f}, {hi:+.3f}]' if sign else f'{m:.3f} [{lo:.3f}, {hi:.3f}]')


# ------------------------------------------------------------------ A. ablation
def ablation():
    S = {}   # (set, disease, rep) -> pooled qscore array over all households;  fold-level rows
    rows = []
    for r, f in splits():
        z2 = np.load(P1 / 'E2_featuresets' / f'r{r}_f{f}.npz'); z11 = np.load(R2 / 'E11_ablation' / f'r{r}_f{f}.npz'); te = z2['test_idx']; assert (te == z11['test_idx']).all()
        for d, dn in enumerate(DIS):
            sc = {'FS1': (z2['oof'][:, d, 0], z2['test'][:, d, 0]), 'FS5': (z2['oof'][:, d, 1], z2['test'][:, d, 1])}
            for s, name in enumerate(ABLATION): sc[name] = (z11['oof'][:, d, s], z11['test'][:, d, s])
            for name, (so, st) in sc.items():
                y = Y[te, d]; rows.append({'rep': r, 'fold': f, 'disease': dn, 'set': name, 'ROC_AUC': auc(y, st), 'PR_AUC': pr_auc(y, st), 'within_year_AUC': within_year_auc(y, st, YR[te])[0]})
                S.setdefault((name, dn, r), np.full(len(Y), np.nan))[te] = ecdf_rank(so, st)
    F = pd.DataFrame(rows); F.to_csv(R2 / 'E11_ablation' / 'ablation_fold_metrics_v1.csv', index=False)
    tab, inc = [], []
    for dn in DIS:
        d = DIS.index(dn); y = Y[:, d]
        def pooled(name, ix): return np.mean([auc(y[ix], S[(name, dn, r)][ix]) for r in (1, 2, 3)])
        bvals = {name: np.array([pooled(name, ix) for ix in draws()]) for name in SETS}
        for name in SETS:
            q = F[(F.disease == dn) & (F.set == name)]; est = pooled(name, np.arange(len(y)))
            tab.append({'Disease': dn, 'Feature set': name, 'Features': NFEAT[name], 'ROC-AUC fold mean (SD)': f'{q.ROC_AUC.mean():.3f} ({q.ROC_AUC.std():.3f})',
                        'ROC-AUC pooled [95% CI]': ci_str(est, *np.quantile(bvals[name], [0.025, 0.975])), 'PR-AUC fold mean (no-skill)': f'{q.PR_AUC.mean():.3f} ({y.mean():.3f})',
                        'Within-year AUC fold mean': f'{q.within_year_AUC.mean():.3f}', '_roc': q.ROC_AUC.mean(), '_wy': q.within_year_AUC.mean()})
        for a, b in zip(SETS[:-1], SETS[1:]):
            qa = F[(F.disease == dn) & (F.set == a)].sort_values(['rep', 'fold']); qb = F[(F.disease == dn) & (F.set == b)].sort_values(['rep', 'fold'])
            dd = qb.ROC_AUC.values - qa.ROC_AUC.values; m, lo, hi, p = corrected_t(dd, 1, RATIO, 15); dw = qb.within_year_AUC.values - qa.within_year_AUC.values; mw, lw, hw, pw = corrected_t(dw, 1, RATIO, 15)
            bd = bvals[b] - bvals[a]; est = pooled(b, np.arange(len(y))) - pooled(a, np.arange(len(y)))
            inc.append({'Step': f'{b} - {a}', 'Added group': {'FS2': 'Climate indices (6)', 'FS3': 'Coordinates (4)', 'FS4': 'Neighbourhood and cross-disease rates (18)', 'FS5': 'Sampling density (6)'}[b], 'Disease': dn,
                        'Fold mean [corrected-t 95% CI]': ci_str(m, lo, hi, True), 'Folds > 0': f'{(dd > 0).sum()}/15', 'p': p, 'Pooled [bootstrap 95% CI]': ci_str(est, *np.quantile(bd, [0.025, 0.975]), sign=True),
                        'Within-year fold mean [corrected-t 95% CI]': ci_str(mw, lw, hw, True), '_m': m, '_lo': lo, '_hi': hi})
    T = pd.DataFrame(tab); I = pd.DataFrame(inc); I['Holm p (12 tests)'] = holm(I.p.values); I['p'] = I.p.map(lambda v: f'{v:.3f}'); I['Holm p (12 tests)'] = I['Holm p (12 tests)'].map(lambda v: f'{v:.3f}')
    T.drop(columns=[c for c in T if c.startswith('_')]).to_csv(TAB / 'R2T1_ablation_performance_v1.csv', index=False); I.drop(columns=[c for c in I if c.startswith('_')]).to_csv(TAB / 'R2T2_ablation_increments_v1.csv', index=False)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw={'width_ratios': [1, 1.25]})
    ax = axs[0]
    for dn, mk, c in zip(DIS, ('o', 's', '^'), ('#7a7a74', '#0072B2', '#D55E00')):
        s = T[T.Disease == dn]; ax.plot(range(5), s._roc, marker=mk, color=c, lw=1.2, ms=4.5, label=f'{dn} overall'); ax.plot(range(5), s._wy, marker=mk, color=c, lw=1, ls='--', mfc='white', ms=4.5, label=f'{dn} within-year')
    ax.set_xticks(range(5)); ax.set_xticklabels(SETS); ax.axhline(0.5, color='#999', lw=0.6, ls=':'); ax.set_ylabel('ROC-AUC (mean of 15 folds)'); ax.set_title('(a) Progressive feature sets', loc='left')
    ax.legend(fontsize=6.5, ncol=2, loc='upper left', bbox_to_anchor=(0, -0.18))
    ax = axs[1]; steps = list(dict.fromkeys(I.Step)); yb = np.arange(len(steps))
    for j, (dn, c) in enumerate(zip(DIS, ('#7a7a74', '#0072B2', '#D55E00'))):
        s = I[I.Disease == dn].set_index('Step').reindex(steps); off = (j - 1) * 0.25
        ax.errorbar(s._m, yb + off, xerr=[s._m - s._lo, s._hi - s._m], fmt='o', color=c, ms=4, capsize=2, lw=1, label=dn)
    ax.axvline(0, color='#555', lw=0.6); ax.set_yticks(yb); ax.set_yticklabels([f'{s}\n+{I[I.Step == s]["Added group"].iloc[0].split(" (")[0].lower()}' for s in steps], fontsize=7); ax.invert_yaxis()
    ax.set_xlabel('ROC-AUC change (corrected-t 95% CI)'); ax.set_title('(b) Contribution of each added group', loc='left'); ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout(); save(fig, 'R2Fig1_ablation_v1'); return T, I


# ------------------------------------------------------------------ B. SHAP and partial dependence
def group_of(n, d):
    if n in ('Altitude', 'Temperature', 'Precipitation', 'Relative_humidity', 'Wind_speed'): return 'Environmental'
    if n in ('Lat', 'Lon', 'LatPlusLon', 'LatMinusLon'): return 'Coordinates'
    if n.startswith(('dens', 'knndist')): return 'Sampling density'
    if n.startswith(('kern', 'knn')): return 'Own-disease rates' if n.endswith(f'_{d}') else 'Cross-disease rates'
    return 'Climate indices'
PRETTY = {'Relative_humidity': 'Relative humidity', 'Wind_speed': 'Wind speed', 'SL_temperature': 'Sea-level temperature', 'DeMartonne': 'De Martonne index', 'Temp_x_RH': 'Temperature x RH',
          'Precip_x_RH': 'log precipitation x RH', 'log_Precip': 'log precipitation', 'Wind_x_RH': 'Wind x RH', 'LatPlusLon': 'Lat + Lon', 'LatMinusLon': 'Lat - Lon'}
def pretty(n):
    if n.startswith('kern'): b, k = n[4:].split('_'); return f'{DIS[int(k)]} kernel rate {b} km'
    if n.startswith('knndist'): return f'distance to {n[7:]}th nearest'
    if n.startswith('knn'): b, k = n[3:].split('_'); return f'{DIS[int(k)]} rate, {b} nearest'
    if n.startswith('dens'): return f'household density {n[4:]} km'
    return PRETTY.get(n, n)


def shap_pdp():
    Z = [np.load(R2 / 'E12_shap_pdp' / f'r1_f{f}.npz') for f in range(5)]; names = list(Z[0]['names']); rows, grows = [], []
    for d, dn in enumerate(DIS):
        sv = np.concatenate([z[f'shap_{d}'] for z in Z]); mabs = np.abs(sv).mean(0)
        for j, n in enumerate(names): rows.append({'disease': dn, 'feature': n, 'label': pretty(n), 'group': group_of(n, d), 'mean_abs_SHAP': mabs[j], 'impurity_importance': np.mean([z[f'imp_{d}'][j] for z in Z])})
        for g in GROUP_COL:
            cols = [j for j, n in enumerate(names) if group_of(n, d) == g]; grows.append({'disease': dn, 'group': g, 'features': len(cols), 'mean_abs_SHAP_sum': float(np.abs(sv[:, cols].sum(1)).mean())})
    S = pd.DataFrame(rows); G = pd.DataFrame(grows); S.to_csv(TAB / 'R2T3_shap_feature_importance_v1.csv', index=False)
    E8 = pd.read_csv(P1 / 'tables' / 'TableS_grouped_permutation_importance_v1.csv'); E8['g'] = E8.group.str.split(' \(').str[0]
    G = G.merge(E8[['disease', 'g', 'mean_AUC_drop']].rename(columns={'g': 'group', 'mean_AUC_drop': 'permutation_AUC_drop'}), on=['disease', 'group'], how='left')
    rho = {dn: stats.spearmanr(G[G.disease == dn].mean_abs_SHAP_sum, G[G.disease == dn].permutation_AUC_drop).statistic for dn in DIS}
    G.to_csv(TAB / 'R2T4_shap_group_importance_v1.csv', index=False)
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.4))
    for k, (ax, dn) in enumerate(zip(axs, DIS)):
        s = S[S.disease == dn].nlargest(10, 'mean_abs_SHAP').iloc[::-1]
        ax.barh(range(len(s)), s.mean_abs_SHAP, color=[GROUP_COL[g] for g in s.group], height=0.7); ax.set_yticks(range(len(s))); ax.set_yticklabels(s.label, fontsize=6.8)
        ax.set_title(f'({"abc"[k]}) {dn}', loc='left'); ax.set_xlabel('mean |SHAP|')
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COL.values()]; fig.legend(hs, list(GROUP_COL), loc='lower center', ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, 'R2Fig2_shap_top_features_v1')
    fig, axs = plt.subplots(3, 3, figsize=(7.4, 6.2))
    for d, dn in enumerate(DIS):
        top = S[S.disease == dn].nlargest(3, 'mean_abs_SHAP').feature.tolist()
        for k, n in enumerate(top):
            ax = axs[d, k]; j = names.index(n)
            for z in Z: ax.plot(z[f'pd_grid_{d}'][j], z[f'pd_avg_{d}'][j], color='#0072B2', lw=0.8, alpha=0.6)
            ax.set_title(f'{dn}: {pretty(n)}', loc='left', fontsize=8); ax.set_ylabel('Mean predicted probability' if k == 0 else '', fontsize=7.5); ax.tick_params(labelsize=7)
    fig.text(0.5, -0.01, 'Feature value (training-data range, 5th to 95th percentile); one line per development fold of repeat 1', ha='center', fontsize=7.5)
    fig.tight_layout(); save(fig, 'R2Fig3_partial_dependence_v1'); return S, G, rho


# ------------------------------------------------------------------ C1. area of applicability (surveyed locations)
def aoa():
    A = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R2 / 'E13_aoa' / '*.csv')))]); rows = []
    P = pd.read_pickle(P1 / 'predictions' / 'dev_predictions_v2.pkl')
    for feat in ('FS1', 'FS5'):
        for dn in DIS:
            d = DIS.index(dn); s = A[(A.features == feat) & (A.disease == dn)]
            q = P[(P.model == feat) & (P.disease == dn)]; qs = {}
            for _, row in q.iterrows(): qs.update({(row.rep, int(h)): v for h, v in zip(row.test_idx, row.qscore)})
            s = s.assign(q=[qs[(r, h)] for r, h in zip(s.rep, s.household)], y=Y[s.household.values, d])
            ins, out = s[s.inside_AOA], s[~s.inside_AOA]
            def auc_by_rep(t): return np.nanmean([auc(t[t.rep == r].y, t[t.rep == r].q) if t[t.rep == r].y.nunique() == 2 else np.nan for r in (1, 2, 3)])
            rows.append({'Features': feat, 'Disease': dn, 'Test households inside AOA (%)': f'{100 * s.inside_AOA.mean():.1f}', 'Locations ever outside AOA': int(s[~s.inside_AOA].household.map(dict(enumerate(LOC))).nunique()),
                         'Median DI': f'{s.DI.median():.3f}', 'Median threshold': f'{s.threshold.median():.3f}', 'ROC-AUC inside AOA': f'{auc_by_rep(ins):.3f}', 'ROC-AUC outside AOA': f'{auc_by_rep(out):.3f}' if len(out) and out.y.nunique() == 2 else 'n/a',
                         'Households outside (positives)': f'{len(out) // 3} ({int(out.y.sum()) // 3})', 'Spearman DI vs |error|': f'{stats.spearmanr(s.DI, np.abs(s.y - s.q)).statistic:+.3f}'})
    T = pd.DataFrame(rows); T.to_csv(TAB / 'R2T5_area_of_applicability_v1.csv', index=False)
    Ld = df.groupby('loc')[['Longitude', 'Latitude']].first()
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.9), sharey=True)
    for k, (ax, dn) in enumerate(zip(axs, DIS)):
        s = A[(A.features == 'FS5') & (A.disease == dn)].assign(loc=lambda t: LOC[t.household.values]); g = s.groupby('loc').agg(DI=('DI', 'mean'), thr=('threshold', 'mean'), out=('inside_AOA', lambda v: (~v).mean()))
        xy = Ld.loc[g.index]; sc = ax.scatter(xy.Longitude, xy.Latitude, c=g.DI / g.thr, cmap='viridis', vmin=0, vmax=1.5, s=12, edgecolor='none')
        o = g.out > 0; ax.scatter(xy.Longitude[o], xy.Latitude[o], s=34, facecolor='none', edgecolor='#D55E00', lw=1)
        ax.set_title(f'({"abc"[k]}) {dn}', loc='left'); ax.set_xlabel('Longitude'); ax.set_aspect('equal')
        if k == 0: ax.set_ylabel('Latitude')
    cb = fig.colorbar(sc, ax=axs, shrink=0.8, pad=0.01); cb.set_label('DI / AOA threshold (mean over repeats)', fontsize=7.5)
    fig.text(0.01, -0.02, 'Orange rings: locations outside the area of applicability in at least one repeat. Points are surveyed locations only; values are not risk.', fontsize=7)
    save(fig, 'R2Fig4_area_of_applicability_FS5_v1'); return T


# ------------------------------------------------------------------ D. confidence intervals for all metrics (pooled per repeat, cluster bootstrap)
def logistic(x, y, offset=None):
    X = np.column_stack([np.ones_like(x), x]) if offset is None else np.ones((len(x), 1)); b = np.zeros(X.shape[1]); off = 0 if offset is None else offset
    for _ in range(50):
        eta = X @ b + off; p = 1 / (1 + np.exp(-eta)); W = p * (1 - p); g = X.T @ (y - p); H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(X.shape[1]); st = np.linalg.solve(H, g); b += st
        if np.abs(st).max() < 1e-8: break
    return b


def metric_cis():
    P = pd.read_pickle(P1 / 'predictions' / 'dev_predictions_v2.pkl'); models = ['FS1', 'FS5', '(a) Rank average', '(b) Inner-selected single']; rows = []; calib = {}
    for dn in DIS:
        d = DIS.index(dn); y = Y[:, d]
        for m in models:
            arr = {}
            for r in (1, 2, 3):
                q = P[(P.model == m) & (P.disease == dn) & (P.rep == r)]; qs = np.full(len(y), np.nan); pr = qs.copy(); yh = qs.copy()
                for _, row in q.iterrows(): qs[row.test_idx] = row.qscore; pr[row.test_idx] = row.prob; yh[row.test_idx] = (row.prob >= row.threshold).astype(float)
                arr[r] = (qs, np.clip(pr, 1e-6, 1 - 1e-6), yh)
            def stat(ix):
                out = []
                for r in (1, 2, 3):
                    qs, pr, yh = arr[r]; yy = y[ix]; p = pr[ix]; h = yh[ix]; lp = np.log(p / (1 - p))
                    tp = ((h == 1) & (yy == 1)).sum(); tn = ((h == 0) & (yy == 0)).sum(); fp = ((h == 1) & (yy == 0)).sum(); fn = ((h == 0) & (yy == 1)).sum()
                    se, sp = tp / max(tp + fn, 1), tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1); den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
                    slope = logistic(lp, yy)[1] if np.ptp(lp) > 1e-9 else np.nan
                    out.append([pr_auc(yy, qs[ix]), np.mean((p - yy) ** 2), logistic(lp, yy, offset=lp)[0], slope, se, sp, (se + sp) / 2, 2 * prec * se / max(prec + se, 1e-12), (tp * tn - fp * fn) / den if den else 0.0])
                return np.nanmean(np.array(out, float), 0)
            est = stat(np.arange(len(y))); bv = np.array([stat(ix) for ix in draws()]); lo, hi = np.nanquantile(bv, 0.025, axis=0), np.nanquantile(bv, 0.975, axis=0)
            labels = ['PR-AUC', 'Brier', 'Calibration intercept', 'Calibration slope', 'Sensitivity', 'Specificity', 'Balanced accuracy', 'F1', 'MCC']
            rec = {'Disease': dn, 'Model': m, 'Prevalence': f'{y.mean():.3f}'}
            for k, lab in enumerate(labels): rec[lab] = f'{est[k]:.3f} [{lo[k]:.3f}, {hi[k]:.3f}]'
            rows.append(rec)
            if m in ('FS5', '(a) Rank average'):
                pr = arr[1][1]; edges = np.quantile(pr, np.linspace(0, 1, 11)); kbin = np.clip(np.digitize(pr, edges[1:-1]), 0, 9)
                def curve(ix): return np.array([[pr[ix][kbin[ix] == j].mean(), y[ix][kbin[ix] == j].mean()] for j in range(10)])
                calib[(dn, m)] = (curve(np.arange(len(y))), np.array([curve(ix)[:, 1] for ix in draws()[:B]]))
    T = pd.DataFrame(rows); T.to_csv(TAB / 'R2T6_metrics_with_CI_v1.csv', index=False)
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.8))
    for k, (ax, dn) in enumerate(zip(axs, DIS)):
        for m, c in (('FS5', '#0072B2'), ('(a) Rank average', '#D55E00')):
            cur, bvals = calib[(dn, m)]; lo, hi = np.nanquantile(bvals, [0.025, 0.975], axis=0)
            ax.fill_between(cur[:, 0], lo, hi, color=c, alpha=0.18, lw=0); ax.plot(cur[:, 0], cur[:, 1], 'o-', color=c, ms=3, lw=1, label=m)
        ax.plot([0, 1], [0, 1], ':', color='#999', lw=0.8); ax.set_xlim(0, 0.75); ax.set_ylim(0, 0.75); ax.set_title(f'({"abc"[k]}) {dn}', loc='left'); ax.set_xlabel('Mean predicted probability')
        if k == 0: ax.set_ylabel('Observed proportion'); ax.legend(fontsize=7, loc='upper left')
    fig.text(0.99, -0.03, 'Deciles of predicted probability (repeat 1); bands: 95% location-cluster bootstrap', ha='right', fontsize=7, color='#555')
    fig.tight_layout(); save(fig, 'R2Fig5_calibration_with_bands_v1'); return T


# ------------------------------------------------------------------ E. campaign identification
def campaign_identification():
    C = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R2 / 'E14_campaign_identification' / '*.csv')))]); rows = []
    for name in ('FS1', 'Neighbourhood rates only', 'FS5'):
        s = C[C.features == name].sort_values(['rep', 'fold']); m, lo, hi, _ = corrected_t(s.macro_AUC_year.values - 0.5, 1, RATIO, 15)
        rows.append({'Features': name, 'Macro ROC-AUC for survey year [corrected-t 95% CI]': ci_str(m + 0.5, lo + 0.5, hi + 0.5), 'Accuracy (5 classes)': f'{s.accuracy.mean():.3f}',
                     **{f'AUC {y}': f'{s[f"AUC_{y}"].mean():.3f}' for y in (2012, 2014, 2015, 2018, 2019)}})
    a = C[C.features == 'Neighbourhood rates only'].sort_values(['rep', 'fold']).macro_AUC_year.values; b = C[C.features == 'FS1'].sort_values(['rep', 'fold']).macro_AUC_year.values
    m, lo, hi, p = corrected_t(a - b, 1, RATIO, 15); T = pd.DataFrame(rows); T.to_csv(TAB / 'R2T7_campaign_identification_v1.csv', index=False); return T, (m, lo, hi, p)


# ------------------------------------------------------------------ F. ensemble without the group-balanced CatBoost model
def ensemble_without_gb(j_gb=13):
    rows = []
    for i in range(5, 20):
        z = np.load(P1 / 'E3_library23' / f'split_{i:02d}.npz'); te = z['test_idx']; keep = [m for m in range(23) if m != j_gb]
        for d, dn in enumerate(DIS):
            Po, Pt = z['oof'][:, d, :], z['test'][:, d, :]
            u23 = np.column_stack([ecdf_rank(Po[:, m], Pt[:, m]) for m in range(23)]).mean(1); u22 = np.column_stack([ecdf_rank(Po[:, m], Pt[:, m]) for m in keep]).mean(1)
            rows.append({'split': i, 'disease': dn, 'AUC23': auc(Y[te, d], u23), 'AUC22': auc(Y[te, d], u22), 'WY23': within_year_auc(Y[te, d], u23, YR[te])[0], 'WY22': within_year_auc(Y[te, d], u22, YR[te])[0]})
    R = pd.DataFrame(rows); out = []
    for dn in DIS:
        s = R[R.disease == dn]; m, lo, hi, p = corrected_t(s.AUC22 - s.AUC23, 1, RATIO, 15); mw, lw, hw, _ = corrected_t(s.WY22 - s.WY23, 1, RATIO, 15)
        out.append({'Disease': dn, 'Rank average, 23 models': f'{s.AUC23.mean():.3f}', 'Without group-balanced CatBoost (22)': f'{s.AUC22.mean():.3f}', 'Difference [corrected-t 95% CI]': ci_str(m, lo, hi, True),
                    'Max fold |difference|': f'{(s.AUC22 - s.AUC23).abs().max():.4f}', 'Within-year difference [corrected-t 95% CI]': ci_str(mw, lw, hw, True)})
    T = pd.DataFrame(out); T.to_csv(TAB / 'R2T8_ensemble_without_groupbalanced_v1.csv', index=False); return T


# ------------------------------------------------------------------ G. train-test distance distributions
def distance_distributions():
    from paper1.experiments import _blocks
    D = NB.D; rows = []
    def add(scheme, te, tr):
        tl = np.unique(LOC[tr]); dist = D[np.ix_(LOC[te], tl)].min(1); rows.extend({'scheme': scheme, 'distance_km': v} for v in dist)
    for f in range(5): add('Random 5-fold CV', np.where(df.random_fold.values == f)[0], np.where(df.random_fold.values != f)[0])
    for r in (1, 2, 3):
        fo = df[f'spatial_fold_r{r}'].values
        for f in range(5):
            te = np.where(fo == f)[0]; tr = np.where(fo != f)[0]; add('Spatial 81 km', te, tr)
            dmin = D[:, np.unique(LOC[te])].min(1)
            for bkm in (50, 100): add(f'Spatial 81 km + {bkm} km buffer', te, tr[dmin[LOC[tr]] > bkm])
    for bkm in (120.0, 160.0):
        bl = _blocks(df, bkm)
        for r in (1, 2, 3):
            fo = bl[r][1]
            for f in range(5): add(f'Spatial {int(bkm)} km', np.where(fo == f)[0], np.where(fo != f)[0])
    Dd = pd.DataFrame(rows); order = ['Random 5-fold CV', 'Spatial 81 km', 'Spatial 120 km', 'Spatial 160 km', 'Spatial 81 km + 50 km buffer', 'Spatial 81 km + 100 km buffer']
    T = Dd.groupby('scheme').distance_km.agg(**{'Same location in training (%)': lambda v: f'{100 * (v == 0).mean():.1f}', 'Median (km)': lambda v: f'{v.median():.1f}',
                                               'IQR (km)': lambda v: f'{v.quantile(0.25):.1f}-{v.quantile(0.75):.1f}', '95th percentile (km)': lambda v: f'{v.quantile(0.95):.1f}', 'Test households (all folds)': 'size'}).reindex(order).reset_index()
    T.to_csv(TAB / 'R2T9_train_test_distances_v1.csv', index=False)
    fig, ax = plt.subplots(figsize=(5.6, 3.0)); cols = ['#D55E00', '#0072B2', '#4292c6', '#9ecae1', '#7a7a74', '#b9b9b2']; lss = ['-', '-', '--', ':', '-.', '-.']
    for s, c, ls in zip(order, cols, lss):
        v = np.sort(Dd[Dd.scheme == s].distance_km.values); ax.plot(np.maximum(v, 0.5), np.arange(1, len(v) + 1) / len(v), color=c, ls=ls, lw=1.4, label=s)
    ax.set_xscale('log'); ax.set_xlabel('Distance from test household to nearest training location (km, log scale; 0 shown at 0.5)'); ax.set_ylabel('Cumulative share of test households')
    ax.legend(fontsize=7, loc='lower right'); fig.tight_layout(); save(fig, 'R2Fig6_train_test_distances_v1'); return T


# ------------------------------------------------------------------ E10b reproduces E10 (fold-level within-campaign AUCs)
def check_e10b():
    E = pd.concat([pd.read_csv(f) for f in glob.glob(str(REV / 'E10_campaign_cv' / 'r*_f*.csv'))]); mx = 0.0
    for r, f in splits():
        z = np.load(R2 / 'E10b_campaign_predictions' / f'r{r}_f{f}.npz'); te = z['test_idx']
        for _, row in E[(E.rep == r) & (E.fold == f)].iterrows():
            d = DIS.index(row.disease); m = YR[te] == row.year
            for s, name in enumerate(('FS1', 'FS5')):
                v = row[f'campaign_{name}_ROC_AUC']
                if np.isfinite(v): mx = max(mx, abs(auc(Y[te[m], d], z['campaign_test'][m, d, s]) - v))
    return mx


if __name__ == '__main__':
    out = {}
    T, I = ablation(); print(T.drop(columns=[c for c in T if c.startswith('_')]).to_string()); print(I.drop(columns=[c for c in I if c.startswith('_')]).to_string())
    S, G, rho = shap_pdp(); out['shap_vs_permutation_spearman'] = rho; print(G.round(4).to_string())
    print(aoa().to_string()); print(metric_cis().to_string())
    C, diff = campaign_identification(); out['rates_minus_FS1_year_AUC'] = diff; print(C.to_string(), diff)
    print(ensemble_without_gb().to_string()); print(distance_distributions().to_string())
    out['E10b_max_abs_difference_vs_E10'] = check_e10b(); json.dump(out, open(R2 / 'rev2_analysis_summary_v1.json', 'w'), indent=1, default=float); print(out)
