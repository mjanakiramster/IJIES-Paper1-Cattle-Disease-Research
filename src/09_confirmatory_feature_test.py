"""Confirmatory test (opens the sealed partition ONCE) for the hypotheses registered in decision_log.json
['confirmatory_preregistration']: H1 FS5 > FS1 for FMD and LSD; H2 FS5 vs (a) simple rank average (two-sided estimate).
SR-SGO is NOT evaluated here (it failed the Stage 6 gate).  Run from the project root:  python src/09_confirmatory_feature_test.py"""
import sys, json, time, warnings, shutil, datetime
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.data import load, labels, file_hashes, DIS, ENV, SEED
from common.features import climate_coords, Neighbourhood, CLIMATE_NAMES
from common.fitness import FeatureBuilder
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
OUT = Path('results/stage9_confirmatory_v1'); (OUT / 'figures').mkdir(parents=True, exist_ok=False)
LOGF = Path('configs/decision_log.json'); LOG = json.load(open(LOGF)); t_start = time.time()

# ---------- 0. guards: pre-registration exists, sealed file unchanged ----------
assert 'confirmatory_preregistration' in LOG, 'hypotheses must be registered before opening'
CF = Path('results/stage2_v1/confirmatory_folds.csv')
assert file_hashes(CF)['sha256'] == LOG['confirmatory_partition']['sha256'], 'confirmatory file changed since it was sealed'
opened_at = datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds')
CONF = pd.read_csv(CF)
df = load('data/raw'); Y = labels(df); FB = FeatureBuilder(df, Y, Neighbourhood(df, Y), climate_coords(df))
assert (CONF.ID.values == df.ID.values).all()

# ---------- 1. models ----------
base = {'coordinates': False, 'kernel_rates': False, 'knn_rates': False, 'idw_rates': False, 'density_terms': False, 'cross_disease_rates': False,
        'bandwidths': (50.0, 150.0, 400.0), 'ks': (5, 15, 40), 'k': 15, 'prior': 1.0, 'idw_power': 2.0, 'min_samples_leaf': 10, 'max_features': 'sqrt', 'complexity': 0}
FS = {'FS1': {**base, 'climate': ENV},
      'FS5': {**base, 'climate': CLIMATE_NAMES, 'coordinates': True, 'kernel_rates': True, 'knn_rates': True, 'density_terms': True, 'cross_disease_rates': True}}

src = open('data/reference/run3_cattle_v5_pipeline_v3.py').read()
src = src[:src.index("t0 = time.time()\nRES = sorted(")].replace("DATA = Path('data')", "DATA = Path('data/raw')").replace("OUT = next_free_dir('results')", "OUT = next_free_dir('_run3_setup_scratch')")
R3 = {'__name__': 'run3'}; exec(compile(src, 'run3_cattle_v5_pipeline_v3.py', 'exec'), R3); shutil.rmtree(R3['OUT'], ignore_errors=True)
assert (R3['df'].ID.values == CONF.ID.values).all()
for a in (1, 2):
    col = f'cblock_c{a}'; R3['df'][col] = CONF[col].values
    R3['BLOCK_LOC'][col] = R3['df'].groupby('loc')[col].first().sort_index().to_numpy(dtype=str)

def lgbm(): return LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05, min_child_samples=20, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, reg_lambda=1.0, n_jobs=1, verbose=-1, random_state=SEED)
def svm(): return make_pipeline(StandardScaler(), SVC(C=1.0, gamma='scale'))
def lib23(fit_idx, pred_idx, col):
    P = R3['fit_predict_lib'](fit_idx, pred_idx, col); Xf, Xp = R3['featureset']('ELK', fit_idx, pred_idx, col); extra = np.zeros((len(pred_idx), 3, 2))
    for d in range(3):
        y = Y[fit_idx, d]; extra[:, d, 0] = lgbm().fit(Xf, y).predict_proba(Xp)[:, 1]; extra[:, d, 1] = svm().fit(Xf, y).decision_function(Xp)
    return np.concatenate([P, extra], axis=2)
def ecdf_rank_average(Po, Pt):
    n = len(Po); U = np.empty_like(Pt)
    for m in range(Po.shape[1]):
        s = np.sort(Po[:, m]); U[:, m] = (np.searchsorted(s, Pt[:, m], 'left') + np.searchsorted(s, Pt[:, m], 'right')) / (2 * n)
    return U.mean(1)

SPL = [(a, f) for a in (1, 2) for f in range(5)]
def one_split(i):
    warnings.filterwarnings('ignore')
    a, f = SPL[i]; col = f'cblock_c{a}'; fold = CONF[f'cfold_c{a}'].values; te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
    out = {'i': i, 'rep': a, 'fold': f, 'te': te}
    for name, spec in FS.items():
        for d in range(3):
            Xf, Xp, _ = FB.build(spec, tr, te, df[col].values if col in df else CONF[col].values, d)
            out[(name, d)] = ExtraTreesClassifier(300, min_samples_leaf=10, max_features='sqrt', n_jobs=1, random_state=SEED).fit(Xf, Y[tr, d]).predict_proba(Xp)[:, 1]
    Pt = lib23(tr, te, col); fo = R3['inner_folds'](tr, col, SEED + 500 + i, 'spatial'); Po = np.zeros((len(tr), 3, Pt.shape[2]))
    for k in range(5):
        q = fo == k; Po[q] = lib23(tr[~q], tr[q], col)
    for d in range(3): out[('(a)', d)] = ecdf_rank_average(Po[:, d, :], Pt[:, d, :])
    return out

t0 = time.time()
RES = Parallel(n_jobs=2, backend='threading')(delayed(one_split)(i) for i in range(len(SPL)))
print(f'confirmatory fits: {(time.time() - t0) / 60:.1f} min')

# ---------- 2. metrics ----------
pred_rows, fm = [], []
for r in RES:
    for d, dn in enumerate(DIS):
        yt = Y[r['te'], d]
        for m in ('FS1', 'FS5', '(a)'):
            fm.append({'rep': r['rep'], 'fold': r['fold'], 'disease': dn, 'model': m, 'ROC_AUC': roc_auc_score(yt, r[(m, d)]), 'PR_AUC': average_precision_score(yt, r[(m, d)])})
            pred_rows.append(pd.DataFrame({'rep': r['rep'], 'fold': r['fold'], 'idx': r['te'], 'ID': df.ID.values[r['te']], 'disease': dn, 'model': m, 'score': r[(m, d)], 'y': yt}))
FM = pd.DataFrame(fm); FM.to_csv(OUT / 'confirmatory_fold_metrics.csv', index=False)
PR = pd.concat(pred_rows, ignore_index=True); PR.to_csv(OUT / 'confirmatory_test_predictions.csv.gz', index=False)
LOC = df.Location_id.values
ratio = np.mean([(CONF[f'cfold_c{a}'] == f).mean() / (CONF[f'cfold_c{a}'] != f).mean() for a, f in SPL])
def ct(x):
    x = np.asarray(x, float); v = x.var(ddof=1); return np.nan if v == 0 else 2 * stats.t.sf(abs(x.mean() / np.sqrt((1 / len(x) + ratio) * v)), len(x) - 1)
def boot(d, m1, m2, B=1000):
    A = PR[(PR.disease == d) & (PR.model == m1)].pivot_table(index='idx', columns='rep', values='score')
    Bm = PR[(PR.disease == d) & (PR.model == m2)].pivot_table(index='idx', columns='rep', values='score').loc[A.index]
    y = Y[A.index.values, DIS.index(d)]; lc = LOC[A.index.values]; groups = [np.where(lc == l)[0] for l in np.unique(lc)]; rg = np.random.default_rng(SEED)
    diff = lambda ix: np.mean([roc_auc_score(y[ix], A[c].values[ix]) for c in A.columns]) - np.mean([roc_auc_score(y[ix], Bm[c].values[ix]) for c in Bm.columns])
    vals = [diff(np.concatenate([groups[k] for k in rg.integers(0, len(groups), len(groups))])) for _ in range(B)]
    return diff(np.arange(len(y))), *np.percentile(vals, [2.5, 97.5])
dev3 = pd.read_csv('results/stage3_v1/feature_set_fold_metrics.csv'); dev4 = pd.read_csv('results/stage4_v1/comparator_fold_metrics.csv')
dev = {('FS1', d): dev3[(dev3.disease == d) & (dev3.feature_set == 'FS1 original 5 variables')].ROC_AUC.mean() for d in DIS} | \
      {('FS5', d): dev3[(dev3.disease == d) & (dev3.feature_set == 'FS5 + cross-disease rates')].ROC_AUC.mean() for d in DIS} | \
      {('(a)', d): dev4[(dev4.scheme == 'spatial') & (dev4.disease == d) & dev4.method.str.startswith('(a)')].ROC_AUC.mean() for d in DIS}
S, H = [], {}
for d in DIS:
    piv = FM[FM.disease == d].pivot_table(index=['rep', 'fold'], columns='model', values='ROC_AUC')
    row = {'disease': d, **{f'{m} confirmatory': piv[m].mean() for m in ('FS1', 'FS5', '(a)')}, **{f'{m} development': dev[(m, d)] for m in ('FS1', 'FS5', '(a)')}}
    for m2, tag in (('FS1', 'H1 FS5 - FS1'), ('(a)', 'H2 FS5 - (a)')):
        pt, lo, hi = boot(d, 'FS5', m2); dd = piv['FS5'] - piv[m2]
        row.update({f'{tag}: pooled': pt, f'{tag}: CI_lo': lo, f'{tag}: CI_hi': hi, f'{tag}: mean of folds': dd.mean(), f'{tag}: p_corrected': ct(dd), f'{tag}: folds better': f'{int((dd > 0).sum())}/{len(dd)}'})
    S.append(row)
S = pd.DataFrame(S); S.to_csv(OUT / 'confirmatory_summary.csv', index=False)
h1 = {d: bool(S.set_index('disease').loc[d, 'H1 FS5 - FS1: CI_lo'] > 0) for d in ('FMD', 'LSD')}
RESULT = {'opened_at_utc': opened_at, 'H1_per_disease_CI_above_0': h1, 'H1_confirmed': all(h1.values()),
          'H1_CBPP_exploratory': {k: float(S.set_index('disease').loc['CBPP', k]) for k in ('H1 FS5 - FS1: pooled', 'H1 FS5 - FS1: CI_lo', 'H1 FS5 - FS1: CI_hi')},
          'H2_FS5_minus_a': {d: {k: float(S.set_index('disease').loc[d, f'H2 FS5 - (a): {k}']) for k in ('pooled', 'CI_lo', 'CI_hi')} for d in DIS},
          'runtime_min': round((time.time() - t_start) / 60, 1)}
json.dump(RESULT, open(OUT / 'hypothesis_results.json', 'w'), indent=1)
LOG = json.load(open(LOGF)); LOG['confirmatory_partition']['status'] = f'OPENED once at {opened_at} for the confirmatory_preregistration hypotheses only'
LOG['confirmatory_results'] = RESULT; json.dump(LOG, open(LOGF, 'w'), indent=1, default=str); json.dump(LOG, open(OUT / 'decision_log_snapshot.json', 'w'), indent=1, default=str)

# ---------- 3. figure ----------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, d in zip(axs, DIS):
    s = S.set_index('disease').loc[d]; xx = np.arange(3); bw = 0.38
    devv = [s[f'{m} development'] for m in ('FS1', 'FS5', '(a)')]; conv = [s[f'{m} confirmatory'] for m in ('FS1', 'FS5', '(a)')]
    ax.bar(xx - bw / 2, devv, bw, color='#c3c2b7', label='Development (15 folds)'); ax.bar(xx + bw / 2, conv, bw, color=['#86b6ef', '#2a78d6', '#55554f'], label='Confirmatory (10 sealed folds)')
    for x_, v in zip(xx + bw / 2, conv): ax.text(x_, v + 0.004, f'{v:.3f}', ha='center', fontsize=8.5)
    ax.set_xticks(xx); ax.set_xticklabels(['FS1\n5 original variables', 'FS5\n+ climate, coords,\nneighbourhood, cross-disease', '(a) rank average\n23 models'], fontsize=8)
    ax.set_ylim(0.5, max(conv + devv) + 0.05)
    t = f"FS5 - FS1 = {s['H1 FS5 - FS1: pooled']:+.3f} [{s['H1 FS5 - FS1: CI_lo']:+.3f}, {s['H1 FS5 - FS1: CI_hi']:+.3f}]"
    ax.set_title(f'{d}\n{t}', loc='left', fontsize=9.5)
axs[0].set_ylabel('Mean spatial ROC-AUC'); axs[0].legend(frameon=False, fontsize=8, loc='upper left')
fig.suptitle(f'Confirmatory test (sealed partition opened once). H1 (FS5 > FS1 for FMD and LSD): {"CONFIRMED" if RESULT["H1_confirmed"] else "NOT confirmed"}', x=0.01, ha='left')
fig.tight_layout(); fig.savefig(OUT / 'figures' / 'confirmatory_feature_test_v1.png', dpi=200, bbox_inches='tight')
pd.set_option('display.width', 250); print(S.round(4).T.to_string()); print(json.dumps(RESULT, indent=1))
