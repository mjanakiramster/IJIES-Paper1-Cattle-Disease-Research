"""Stage 5c-A (exploratory, 2026-11-21) — the 15 pairwise relay hybrids on the Stage 5 benchmark check.
Identical setting to Stage 5 part A: 30 agents, 15,000 evaluations, 5 seeds, D = 10 and 30, optimum at the centre and
shifted, same initial populations (rng 1000 + seed + D) and optimiser rng (seed). Base-optimiser results are read from
results/stage5_v1/benchmark_runs.csv (not rerun). Run from the project root: python src/05c_hybrid_benchmarks.py"""
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
SRC = Path(__file__).resolve().parent; sys.path.insert(0, str(SRC))
from common.hybrids import HYBRIDS, PAIRS
from common.benchmarks import make_problem, FUNCS
warnings.filterwarnings('ignore')
ROOT = SRC.parent; OUT = ROOT / 'results/stage5c_hybrids_v1'; (OUT / 'figures').mkdir(parents=True, exist_ok=True)
BASE7 = ['SGO', 'MSGO', 'PSO', 'GA', 'DE', 'GWO', 'HGWOPSO', 'RANDOM']


def bench(opt, fn, D, shifted, seed):
    fit, _ = make_problem(fn, D, shifted)
    X0 = np.random.default_rng(1000 + seed + D).uniform(-1, 1, (30, D))
    _, bf, hist = HYBRIDS[opt](fit, X0, 15000, np.random.default_rng(seed))
    return {'optimiser': opt, 'function': fn, 'D': D, 'shifted': shifted, 'seed': seed, 'final_error': -bf,
            'error_at_1500': -hist[1499], 'error_at_7500': -hist[7499]}


if __name__ == '__main__':
    t0 = time.time()
    rows = Parallel(n_jobs=2)(delayed(bench)(o, fn, D, sh, s) for o in PAIRS for fn in FUNCS for D in (10, 30) for sh in (False, True) for s in range(5))
    H = pd.DataFrame(rows); H.to_csv(OUT / 'hybrid_benchmark_runs.csv', index=False)
    A = pd.concat([pd.read_csv(ROOT / 'results/stage5_v1/benchmark_runs.csv'), H], ignore_index=True)
    A['kind'] = np.where(A.optimiser.isin(PAIRS), 'hybrid', 'base')
    # rank every optimiser within each (function, D, shifted, seed) problem instance: 1 = lowest error
    A['rank'] = A.groupby(['function', 'D', 'shifted', 'seed']).final_error.rank(method='average')
    med = A.groupby(['function', 'D', 'shifted', 'optimiser']).final_error.median().unstack('optimiser'); med.to_csv(OUT / 'benchmark_median_error_all.csv')
    R = A.groupby(['optimiser', 'kind']).agg(mean_rank_all=('rank', 'mean')).reset_index()
    for sh, lab in ((False, 'mean_rank_centre'), (True, 'mean_rank_shifted')):
        R = R.merge(A[A.shifted == sh].groupby('optimiser')['rank'].mean().rename(lab), on='optimiser')
    # does each hybrid beat BOTH of its parents? (median error, per problem, shifted only = the centre-bias-free test)
    sh = A[A.shifted].groupby(['function', 'D', 'optimiser']).final_error.median().unstack('optimiser')
    beat = []
    for p in PAIRS:
        a, b = p.split('-'); w = ((sh[p] < sh[a]) & (sh[p] < sh[b])).sum()
        beat.append({'optimiser': p, 'shifted_problems_beating_both_parents': f'{int(w)}/{len(sh)}',
                     'shifted_problems_beating_best_parent_or_tie': f'{int((sh[p] <= sh[[a, b]].min(axis=1)).sum())}/{len(sh)}'})
    R = R.merge(pd.DataFrame(beat), on='optimiser', how='left').sort_values('mean_rank_shifted'); R.to_csv(OUT / 'benchmark_ranks_all.csv', index=False)
    pd.set_option('display.width', 250); print(R.round(2).to_string(index=False)); print(f'done in {(time.time() - t0) / 60:.1f} min')
