"""Stage 6 — SR-SGO (revised design) on the 15 DEVELOPMENT folds, and the pre-registered go/no-go gate.
(A) choose the shift-invariant improving phase on shifted benchmark functions only (rule logged before running);
(B) SR-SGO runs: 3 x 5 development folds x 3 diseases, 240 evaluations, 2 x 5 inner spatial CV, row-level surrogate,
    anchor = FS5, final model = average of the top-5 configurations (checkpointed, resumable);
(C) comparison with comparators (a)-(d) and FS5, and the gate. The confirmatory partition is not touched.
Run from the project root:  python src/06_sr_sgo.py"""
import os, sys, json, time, warnings, datetime
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from joblib import Parallel, delayed
SRC = Path(__file__).resolve().parent; sys.path.insert(0, str(SRC)); os.environ['PYTHONPATH'] = str(SRC)
from common.optimisers import OPTIMISERS
from common.sr_sgo import sr_sgo_centroid, sr_sgo_attract
from common.benchmarks import make_problem, FUNCS
from common.stage6_worker import run_one
from common.data import load, labels, DIS, SEED
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
ROOT = SRC.parent; OUT = ROOT / 'results/stage6_v1'; (OUT / 'figures').mkdir(parents=True, exist_ok=True); (OUT / 'runs').mkdir(exist_ok=True)
LOGF = ROOT / 'configs/decision_log.json'; PLOG = open(ROOT / 'logs/stage6_v1_progress.log', 'a')
def log(m): PLOG.write(f'{time.strftime("%H:%M:%S")} {m}\n'); PLOG.flush(); print(m, flush=True)

# ---------- A. choose the improving-phase variant on shifted benchmarks only ----------
VF = OUT / 'benchmark_variant_selection.csv'
if not VF.exists():
    OPT = {'SGO': OPTIMISERS['SGO'], 'DE': OPTIMISERS['DE'], 'SR-SGO centroid': sr_sgo_centroid, 'SR-SGO attract': sr_sgo_attract}
    def bench(o, fn, sh, s):
        fit, _ = make_problem(fn, 30, sh); X0 = np.random.default_rng(1000 + s + 30).uniform(-1, 1, (30, 30))
        _, bf, _ = OPT[o](fit, X0, 15000, np.random.default_rng(s)); return {'optimiser': o, 'function': fn, 'shifted': sh, 'seed': s, 'final_error': -bf}
    BV = pd.DataFrame(Parallel(n_jobs=2)(delayed(bench)(o, fn, sh, s) for o in OPT for fn in FUNCS for sh in (False, True) for s in range(5)))
    BV.to_csv(VF, index=False)
BV = pd.read_csv(VF)
med = BV.groupby(['function', 'shifted', 'optimiser']).final_error.median().unstack('optimiser')
ranks = med.xs(True, level='shifted').rank(axis=1).mean()
variant = 'centroid' if ranks['SR-SGO centroid'] <= ranks['SR-SGO attract'] else 'attract'
L = json.load(open(LOGF))
if 'stage6_decisions' not in L:
    L['stage6_decisions'] = {'time_utc': datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds'), 'improving_phase_variant': variant,
                             'mean_rank_on_shifted_functions': ranks.round(3).to_dict(), 'basis': 'benchmark functions only'}
    json.dump(L, open(LOGF, 'w'), indent=1, default=str)
variant = json.load(open(LOGF))['stage6_decisions']['improving_phase_variant']
med.to_csv(OUT / 'benchmark_variant_medians.csv'); log(f'variant chosen: {variant}; mean ranks on shifted functions: {ranks.round(2).to_dict()}')

# ---------- B. SR-SGO on the development folds (checkpointed) ----------
jobs = [(d, r, f) for r in (1, 2, 3) for f in range(5) for d in range(3)]
todo = [j for j in jobs if not (OUT / 'runs' / f'd{j[0]}_r{j[1]}_f{j[2]}.json').exists()]
log(f'SR-SGO runs: {len(jobs)} total, {len(todo)} to do'); t0 = time.time(); n = 0
for _ in Parallel(n_jobs=2, return_as='generator_unordered')(delayed(run_one)(str(ROOT), str(OUT / 'runs'), d, r, f, variant) for d, r, f in todo):
    n += 1
    if n % 3 == 0 or n == len(todo): log(f'  {n}/{len(todo)} runs, {(time.time() - t0) / 60:.1f} min')
RUNS = [json.load(open(OUT / 'runs' / f'd{d}_r{r}_f{f}.json')) for d, r, f in jobs]

# ---------- C. comparison and gate ----------
df = load(ROOT / 'data/raw'); Y = labels(df); LOC = df.Location_id.values
R = pd.DataFrame([{k: v for k, v in x.items() if k not in ('history', 'top_specs', 'top_inner_fitness')} for x in RUNS]); R.to_csv(OUT / 'sr_sgo_runs.csv', index=False)
s3 = pd.read_csv(ROOT / 'results/stage3_v1/feature_set_fold_metrics.csv'); s3 = s3[s3.feature_set == 'FS5 + cross-disease rates']
s4 = pd.read_csv(ROOT / 'results/stage4_v1/comparator_fold_metrics.csv'); s4 = s4[s4.scheme == 'spatial']; s4['fold'] = s4.fold.astype(int)
r3 = pd.read_csv(ROOT / 'data/reference/run3_ensemble_fold_metrics_v1.csv'); r3 = r3[(r3.scheme == 'spatial') & (r3.method == 'RB-HGWOPSO ensemble (proposed)')]; r3['fold'] = r3.fold.astype(int)
best_c = pd.read_csv(ROOT / 'results/stage4_v1/comparators_summary.csv'); best_c = best_c[best_c.method.str.startswith('(c)')].set_index('disease')
fold = R.rename(columns={'outer_ROC_AUC_bagged_top5': 'SR-SGO (proposed, top-5 average)', 'outer_ROC_AUC_single_best': 'SR-SGO single best (ablation)'})[
    ['disease', 'rep', 'fold', 'SR-SGO (proposed, top-5 average)', 'SR-SGO single best (ablation)']]
fold = fold.merge(s3.rename(columns={'ROC_AUC': 'FS5 fixed features (no search)'})[['disease', 'rep', 'fold', 'FS5 fixed features (no search)']], on=['disease', 'rep', 'fold'])
for m, lab in [('(a) Simple rank average, 23 models', '(a) simple rank average'), ('(b) Best single, inner-selected (23 models)', '(b) inner-selected best single')]:
    fold = fold.merge(s4[s4.method == m].rename(columns={'ROC_AUC': lab})[['disease', 'rep', 'fold', lab]], on=['disease', 'rep', 'fold'])
fold = fold.merge(r3.rename(columns={'ROC_AUC': '(d) RB-HGWOPSO run 3'})[['disease', 'rep', 'fold', '(d) RB-HGWOPSO run 3']], on=['disease', 'rep', 'fold'])
fold.to_csv(OUT / 'development_fold_roc_auc.csv', index=False)
METH = [c for c in fold.columns if c not in ('disease', 'rep', 'fold')]
ratio = np.mean([(df[f'spatial_fold_r{r}'] == f).mean() / (df[f'spatial_fold_r{r}'] != f).mean() for r in (1, 2, 3) for f in range(5)])
def ct(x):
    x = np.asarray(x, float); v = x.var(ddof=1); return np.nan if v == 0 else 2 * stats.t.sf(abs(x.mean() / np.sqrt((1 / len(x) + ratio) * v)), len(x) - 1)

# pooled predictions for the bootstrap: SR-SGO from checkpoints, (a) from Stage 4
sr = pd.concat([pd.DataFrame({'idx': np.load(OUT / 'runs' / f'd{d}_r{r}_f{f}_preds.npz')['idx'], 'p': np.load(OUT / 'runs' / f'd{d}_r{r}_f{f}_preds.npz')['p_bag'],
                              'rep': r, 'disease': DIS[d]}) for d, r, f in jobs])
pa = pd.read_csv(ROOT / 'results/stage4_v1/development_test_predictions.csv.gz'); pa = pa[pa.method == '(a) Simple rank average, 23 models'].rename(columns={'score': 'p'})
def boot(d, B=1000):
    A = sr[sr.disease == d].pivot_table(index='idx', columns='rep', values='p'); Bm = pa[pa.disease == d].pivot_table(index='idx', columns='rep', values='p')
    common = A.index.intersection(Bm.index); A, Bm = A.loc[common], Bm.loc[common]; y = Y[common.values, DIS.index(d)]
    lc = LOC[common.values]; groups = [np.where(lc == l)[0] for l in np.unique(lc)]; rg = np.random.default_rng(SEED)
    from sklearn.metrics import roc_auc_score
    diff = lambda ix: np.mean([roc_auc_score(y[ix], A[c].values[ix]) for c in A.columns]) - np.mean([roc_auc_score(y[ix], Bm[c].values[ix]) for c in Bm.columns])
    vals = [diff(np.concatenate([groups[k] for k in rg.integers(0, len(groups), len(groups))])) for _ in range(B)]
    return diff(np.arange(len(y))), *np.percentile(vals, [2.5, 97.5])

summ, gate = [], {}
for d in DIS:
    g = fold[fold.disease == d]; P = 'SR-SGO (proposed, top-5 average)'
    row = {'disease': d, **{m: g[m].mean() for m in METH}, '(c) hindsight-best single': float(best_c.loc[d, 'ROC_AUC_mean_of_folds'])}
    for comp in ['(a) simple rank average', '(b) inner-selected best single', 'FS5 fixed features (no search)', '(d) RB-HGWOPSO run 3', 'SR-SGO single best (ablation)']:
        dd = g[P] - g[comp]; row[f'd vs {comp}'] = dd.mean(); row[f'p vs {comp}'] = ct(dd); row[f'folds better vs {comp}'] = f'{int((dd > 0).sum())}/{len(dd)}'
    pt, lo, hi = boot(d); row.update({'pooled d vs (a)': pt, 'pooled d vs (a) CI_lo': lo, 'pooled d vs (a) CI_hi': hi})
    row['inner fitness: anchor FS5'] = R[R.disease == d].anchor_inner_fitness.mean(); row['inner fitness: best found'] = R[R.disease == d].best_inner_fitness.mean()
    gate[d] = bool(row[P] > row['(a) simple rank average'] and pt > 0); row['gate condition met'] = gate[d]; summ.append(row)
S = pd.DataFrame(summ); S.to_csv(OUT / 'development_summary.csv', index=False)
GATE = {'rule': json.load(open(LOGF))['config']['go_no_go_gate'], 'per_disease': gate, 'diseases_passing': int(sum(gate.values())),
        'decision': 'PROCEED to Stage 8/9' if sum(gate.values()) >= 2 else 'STOP (gate not met) - do not open the confirmatory partition'}
json.dump(GATE, open(OUT / 'go_no_go_gate.json', 'w'), indent=1)
L = json.load(open(LOGF)); L['stage6_gate'] = GATE; json.dump(L, open(LOGF, 'w'), indent=1, default=str); json.dump(L, open(OUT / 'decision_log_snapshot.json', 'w'), indent=1, default=str)

# configuration patterns
cfg = pd.DataFrame([{**{k: x['top_specs'][0][k] for k in ('kernel_rates', 'knn_rates', 'idw_rates', 'density_terms', 'cross_disease_rates', 'min_samples_leaf', 'max_features', 'prior')},
                     'disease': x['disease'], 'n_bandwidths': len(x['top_specs'][0]['bandwidths']), 'n_k': len(x['top_specs'][0]['ks'])} for x in RUNS])
cfg.groupby('disease').mean(numeric_only=True).round(3).to_csv(OUT / 'selected_configuration_patterns.csv')

# figures
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
order = ['SR-SGO (proposed, top-5 average)', 'SR-SGO single best (ablation)', 'FS5 fixed features (no search)', '(a) simple rank average', '(b) inner-selected best single', '(d) RB-HGWOPSO run 3']
cols = ['#eb6834', '#f3a77e', '#2a78d6', '#55554f', '#898781', '#9b59b6']; short = ['SR-SGO\n(top-5)', 'SR-SGO\nsingle', 'FS5\nfixed', '(a) rank\naverage', '(b) inner\nbest', '(d) run-3\nHGWOPSO']
for ax, d in zip(axs, DIS):
    g = fold[fold.disease == d]; m_ = [g[c].mean() for c in order]; se = [g[c].std() / np.sqrt(len(g)) for c in order]
    ax.bar(range(len(order)), m_, yerr=se, color=cols, width=0.7, capsize=3)
    for i, v in enumerate(m_): ax.text(i, v + se[i] + 0.003, f'{v:.3f}', ha='center', fontsize=8)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(short, fontsize=8); ax.set_ylim(min(m_) - 0.05, max(m_) + 0.03)
    ax.set_title(f'{d}: gate {"met" if gate[d] else "not met"}', loc='left')
axs[0].set_ylabel('Mean ROC-AUC, 15 development folds (± SE)')
fig.suptitle(f'Stage 6 — SR-SGO vs comparators on development folds. Gate: {GATE["decision"]}', x=0.01, ha='left', fontsize=10)
fig.tight_layout(); fig.savefig(OUT / 'figures' / 'sr_sgo_vs_comparators_development_v1.png', dpi=200, bbox_inches='tight')
fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, d in zip(axs, DIS):
    A = np.array([x['history'] for x in RUNS if x['disease'] == d]); ax.plot(np.arange(1, A.shape[1] + 1), (A - A[:, :1]).mean(0), color='#eb6834', lw=2)
    ax.axhline((R[R.disease == d].anchor_inner_fitness - A[:, 0]).mean(), color='#2a78d6', ls='--', lw=1, label='FS5 anchor')
    ax.set_title(d, loc='left'); ax.set_xlabel('Fitness evaluations'); ax.legend(frameon=False, fontsize=8)
axs[0].set_ylabel('Best robust fitness gain'); fig.suptitle('SR-SGO convergence on the inner folds (mean of 15 runs)', x=0.01, ha='left')
fig.tight_layout(); fig.savefig(OUT / 'figures' / 'sr_sgo_convergence_v1.png', dpi=200, bbox_inches='tight')
pd.set_option('display.width', 250); log('\n' + S.round(4).T.to_string()); log(json.dumps(GATE, indent=1)); log('Stage 6 done')
