"""Stage 5 — optimisers. (A) correctness check on benchmark functions (plain and shifted), (B) comparison on the real
SR-SGO problem (development repeat 1, 5 outer folds x 3 diseases x 3 seeds, 150 evaluations, 1 x 5 inner spatial CV,
row-level ExtraTrees-100 surrogate), (C) analysis. Resumable: finished runs are checkpointed and skipped on restart.
Run from the project root:  python src/05_optimization_algorithms.py"""
import os, sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from joblib import Parallel, delayed
SRC = Path(__file__).resolve().parent; sys.path.insert(0, str(SRC)); os.environ['PYTHONPATH'] = str(SRC)
from common.optimisers import OPTIMISERS
from common.benchmarks import make_problem, FUNCS
from common.stage5_worker import run_one
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
ROOT = SRC.parent; OUT = ROOT / 'results/stage5_v1'; (OUT / 'figures').mkdir(parents=True, exist_ok=True); (OUT / 'runs').mkdir(exist_ok=True)
LOG = open(ROOT / 'logs/stage5_v1_progress.log', 'a')
def log(msg): LOG.write(f'{time.strftime("%H:%M:%S")} {msg}\n'); LOG.flush(); print(msg, flush=True)
OPTS = ['SGO', 'MSGO', 'PSO', 'GA', 'DE', 'GWO', 'RANDOM']; BENCH_OPTS = OPTS + ['HGWOPSO']
COL = {'SGO': '#2a78d6', 'MSGO': '#0d3f86', 'PSO': '#898781', 'GA': '#c3a000', 'DE': '#1baf7a', 'GWO': '#eb6834', 'RANDOM': '#c3c2b7', 'HGWOPSO': '#9b59b6'}

# ---------- A. benchmark correctness check ----------
BF = OUT / 'benchmark_runs.csv'
if not BF.exists():
    def bench(opt, fn, D, shifted, seed):
        fit, _ = make_problem(fn, D, shifted)
        X0 = np.random.default_rng(1000 + seed + D).uniform(-1, 1, (30, D))
        _, bf, hist = OPTIMISERS[opt](fit, X0, 15000, np.random.default_rng(seed))
        return {'optimiser': opt, 'function': fn, 'D': D, 'shifted': shifted, 'seed': seed, 'final_error': -bf,
                'error_at_1500': -hist[1499], 'error_at_7500': -hist[7499]}
    t0 = time.time()
    rows = Parallel(n_jobs=2)(delayed(bench)(o, fn, D, sh, s) for o in BENCH_OPTS for fn in FUNCS for D in (10, 30) for sh in (False, True) for s in range(5))
    pd.DataFrame(rows).to_csv(BF, index=False); log(f'benchmarks done: {len(rows)} runs in {(time.time() - t0) / 60:.1f} min')
BR = pd.read_csv(BF)

# ---------- B. real problem (checkpointed) ----------
jobs = [(o, d, f, s) for s in range(3) for f in range(5) for d in range(3) for o in OPTS]
todo = [j for j in jobs if not (OUT / 'runs' / f'{j[0]}_d{j[1]}_f{j[2]}_s{j[3]}.json').exists()]
log(f'real-problem runs: {len(jobs)} total, {len(todo)} to do')
t0 = time.time(); done = 0
for res in Parallel(n_jobs=2, return_as='generator_unordered')(delayed(run_one)(str(ROOT), str(OUT / 'runs'), o, d, f, s) for o, d, f, s in todo):
    done += 1
    if done % 7 == 0 or done == len(todo): log(f'  {done}/{len(todo)} runs, {(time.time() - t0) / 60:.1f} min elapsed')
RUNS = [json.load(open(p)) for p in sorted((OUT / 'runs').glob('*.json'))]
assert len(RUNS) == len(jobs), f'only {len(RUNS)} of {len(jobs)} runs present'

# ---------- C. analysis ----------
R = pd.DataFrame([{k: v for k, v in r.items() if k not in ('history', 'best_x', 'spec')} for r in RUNS])
R['spec_n_climate'] = [len(r['spec']['climate']) for r in RUNS]
for k in ('coordinates', 'kernel_rates', 'knn_rates', 'idw_rates', 'density_terms', 'cross_disease_rates', 'min_samples_leaf', 'max_features', 'k', 'prior'):
    R[f'spec_{k}'] = [r['spec'][k] for r in RUNS]
R.to_csv(OUT / 'optimiser_real_runs.csv', index=False)
H = {(r['optimiser'], r['disease'], r['fold'], r['seed']): np.array(r['history']) for r in RUNS}

s3 = pd.read_csv(ROOT / 'results/stage3_v1/feature_set_fold_metrics.csv'); s3 = s3[(s3.rep == 1) & (s3.feature_set == 'FS5 + cross-disease rates')]
s4 = pd.read_csv(ROOT / 'results/stage4_v1/comparator_fold_metrics.csv'); s4 = s4[(s4.scheme == 'spatial') & (s4.rep == 1) & s4.method.str.startswith('(a)')]
refs = {d: {'FS5 fixed features, ExtraTrees-300': s3[s3.disease == d].ROC_AUC.mean(), '(a) simple rank average': s4[s4.disease == d].ROC_AUC.mean()} for d in ['CBPP', 'FMD', 'LSD']}

summ = R.groupby(['disease', 'optimiser']).agg(inner_fitness_mean=('best_inner_fitness', 'mean'), inner_fitness_sd=('best_inner_fitness', 'std'),
                                              outer_ROC_AUC_mean=('outer_ROC_AUC', 'mean'), outer_ROC_AUC_sd=('outer_ROC_AUC', 'std'),
                                              outer_PR_AUC_mean=('outer_PR_AUC', 'mean'), seconds_mean=('seconds', 'mean'), unique_candidates=('unique_candidates', 'mean')).reset_index()
for d in refs:
    for k, v in refs[d].items(): summ.loc[summ.disease == d, f'reference: {k}'] = v
summ.to_csv(OUT / 'optimiser_real_summary.csv', index=False)
overall = R.groupby('optimiser').agg(inner_fitness=('best_inner_fitness', 'mean'), outer_ROC_AUC=('outer_ROC_AUC', 'mean'), seconds=('seconds', 'mean')).reindex(OPTS)

def qalpha(k): return float(stats.studentized_range.ppf(0.95, k, 1e6) / np.sqrt(2))
fr = {}
for metric in ('best_inner_fitness', 'outer_ROC_AUC'):
    M = R.pivot_table(index=['disease', 'fold', 'seed'], columns='optimiser', values=metric)[OPTS]
    ch = stats.friedmanchisquare(*[M[c].values for c in OPTS]); rk = M.rank(axis=1, ascending=False).mean()
    cd = qalpha(len(OPTS)) * np.sqrt(len(OPTS) * (len(OPTS) + 1) / (6 * len(M)))
    fr[metric] = {'friedman_p': float(ch.pvalue), 'nemenyi_cd': cd, 'mean_ranks': rk.round(3).to_dict(), 'blocks': len(M)}
pw = []
for metric in ('best_inner_fitness', 'outer_ROC_AUC'):
    M = R.pivot_table(index=['disease', 'fold', 'seed'], columns='optimiser', values=metric)
    for a in ('SGO', 'MSGO'):
        ps = []
        for b in [o for o in OPTS if o != a]:
            dlt = M[a] - M[b]; p = stats.wilcoxon(dlt).pvalue if (dlt != 0).any() else 1.0
            ps.append({'metric': metric, 'optimiser': a, 'versus': b, 'mean_difference': dlt.mean(), 'wins': int((dlt > 0).sum()), 'ties': int((dlt == 0).sum()), 'losses': int((dlt < 0).sum()), 'p_wilcoxon': p})
        m = len(ps); order = np.argsort([x['p_wilcoxon'] for x in ps]); adj = np.zeros(m)
        for rank_, i in enumerate(order): adj[i] = min(1.0, max(adj[order[rank_ - 1]] if rank_ else 0, (m - rank_) * ps[i]['p_wilcoxon']))
        for x, a_ in zip(ps, adj): x['p_holm'] = a_
        pw += ps
PW = pd.DataFrame(pw); PW.to_csv(OUT / 'pairwise_sgo_msgo_vs_others.csv', index=False)
json.dump({'friedman': fr, 'n_runs': len(R), 'design': 'development repeat 1, 5 outer folds x 3 diseases x 3 seeds; 10 agents, 150 evaluations; 1 x 5 inner spatial CV; row-level ExtraTrees-100 surrogate; same initial population for all optimisers'},
          open(OUT / 'optimiser_statistics.json', 'w'), indent=1)

bsum = BR.groupby(['function', 'D', 'shifted', 'optimiser']).final_error.agg(['mean', 'std', 'median']).reset_index(); bsum.to_csv(OUT / 'benchmark_summary.csv', index=False)

# figures
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, d in zip(axs, ['CBPP', 'FMD', 'LSD']):
    for o in OPTS:
        A = np.array([H[(o, d, f, s)] for f in range(5) for s in range(3)]); A = A - A[:, :1]
        ax.plot(np.arange(1, A.shape[1] + 1), A.mean(0), color=COL[o], lw=2 if o in ('SGO', 'MSGO') else 1.2, label=o)
    ax.set_title(d, loc='left'); ax.set_xlabel('Fitness evaluations'); ax.grid(color='#eeeeea')
axs[0].set_ylabel('Best robust fitness gain over the initial population'); axs[2].legend(frameon=False, fontsize=8, ncol=2)
fig.suptitle('Convergence on the real SR-SGO problem (mean of 5 folds x 3 seeds)', x=0.01, ha='left'); fig.tight_layout(); fig.savefig(OUT / 'figures' / 'convergence_real_problem_v1.png', dpi=200, bbox_inches='tight')

fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=False)
for ax, d in zip(axs, ['CBPP', 'FMD', 'LSD']):
    s = summ[summ.disease == d].set_index('optimiser').reindex(OPTS)
    ax.bar(range(len(OPTS)), s.outer_ROC_AUC_mean, yerr=s.outer_ROC_AUC_sd / np.sqrt(15), color=[COL[o] for o in OPTS], width=0.7, capsize=3)
    for k_, (lab, v) in enumerate(refs[d].items()): ax.axhline(v, color=['#55554f', '#e34948'][k_], ls='--', lw=1.1, label=f'{lab} ({v:.3f})')
    ax.set_xticks(range(len(OPTS))); ax.set_xticklabels(OPTS, fontsize=8.5); ax.set_ylim(min(s.outer_ROC_AUC_mean.min(), min(refs[d].values())) - 0.04, max(s.outer_ROC_AUC_mean.max(), max(refs[d].values())) + 0.03)
    ax.set_title(d, loc='left'); ax.legend(frameon=False, fontsize=7.5, loc='lower right')
axs[0].set_ylabel('Outer-fold ROC-AUC (mean ± SE, 15 runs)'); fig.suptitle('Outer test ROC-AUC of each optimiser\'s best configuration (development repeat 1)', x=0.01, ha='left')
fig.tight_layout(); fig.savefig(OUT / 'figures' / 'outer_auc_by_optimiser_v1.png', dpi=200, bbox_inches='tight')

fig, axs = plt.subplots(2, 4, figsize=(16, 7))
for c, fn in enumerate(FUNCS):
    for r_, sh in enumerate((False, True)):
        ax = axs[r_, c]; s = bsum[(bsum.function == fn) & (bsum.shifted == sh) & (bsum.D == 30)].set_index('optimiser').reindex(BENCH_OPTS)
        ax.bar(range(len(BENCH_OPTS)), np.log10(s['median'].clip(lower=1e-300) + 1e-300), color=[COL[o] for o in BENCH_OPTS], width=0.7)
        ax.set_xticks(range(len(BENCH_OPTS))); ax.set_xticklabels(BENCH_OPTS, rotation=45, fontsize=7.5); ax.set_title(f'{fn}, D=30, {"shifted" if sh else "optimum at centre"}', loc='left', fontsize=9)
    axs[0, c].set_ylabel('log10 median final error' if c == 0 else '')
fig.suptitle('Benchmark check: 15,000 evaluations, 30 agents, 5 seeds (lower = better)', x=0.01, ha='left'); fig.tight_layout(); fig.savefig(OUT / 'figures' / 'benchmark_errors_v1.png', dpi=200, bbox_inches='tight')

pd.set_option('display.width', 220)
log('\n' + overall.round(4).to_string()); log(json.dumps(fr, indent=1))
log('\n' + summ[['disease', 'optimiser', 'inner_fitness_mean', 'outer_ROC_AUC_mean', 'outer_ROC_AUC_sd', 'seconds_mean']].round(4).to_string(index=False))
log('\n' + PW.round(4).to_string(index=False)); log('Stage 5 analysis done')
