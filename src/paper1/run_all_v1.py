"""Runner for all Paper 1 experiments (corrected code). Same functions as the notebook; resumable checkpoints.
Usage from the project root:  python src/paper1/run_all_v1.py [E2 E3 E5 E6 E7 E8]"""
import os, sys, time
from pathlib import Path
SRC = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(SRC)); os.environ['PYTHONPATH'] = str(SRC)
from joblib import Parallel, delayed
from paper1 import experiments as X

def run(name, fn, jobs, n_jobs=2):
    t0 = time.time()
    for k, _ in enumerate(Parallel(n_jobs=n_jobs, return_as='generator_unordered')(delayed(fn)(*j) for j in jobs), 1):
        if k % 10 == 0 or k == len(jobs): print(f'{name}: {k}/{len(jobs)} ({(time.time() - t0) / 60:.1f} min)', flush=True)

if __name__ == '__main__':
    todo = sys.argv[1:] or ['E2', 'E3', 'E7', 'E8', 'E6', 'E5']
    for e in todo:
        if e == 'E2': run('E2', X.e2_split, [(r, f) for r in (1, 2, 3) for f in range(5)])
        if e == 'E3': run('E3', X.e3_split, [(i,) for i in range(5, 20)])
        if e == 'E7': X.e7_temporal(); print('E7 done', flush=True)
        if e == 'E8': run('E8', X.e8_fold, [(r, f) for r in (1, 2, 3) for f in range(5)])
        if e == 'E6': run('E6', X.e6_job, [(c, r, f) for c in X.E6_CONDITIONS for r in (1, 2, 3) for f in range(5)])
        if e == 'E5': run('E5', X.e5_part, [('household', -1)] + [(s, p) for p in range(100) for s in ('household', 'village')])
    print('ALL DONE', flush=True)
