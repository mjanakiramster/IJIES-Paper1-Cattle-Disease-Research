"""Second worker for the from-scratch run (uses the idle second core). Computes the same checkpoints as notebook cell 18,
with the same code in the fresh copy, in REVERSE order, so it meets the notebook (which runs forward) only at the end.
Every job function skips outputs that already exist, so each checkpoint is computed once."""
import os, sys, time
os.chdir('/home/claude/fresh/proj'); sys.path.insert(0, 'src')
from paper1.core import setup, dev_splits
from paper1 import experiments as X, rev_experiments_v1 as RX, rev2_experiments_v1 as R2X
t0 = time.time()
def log(m): print(f'{(time.time() - t0) / 60:6.1f} min  {m}', flush=True)
for f_ in (4, 3, 2, 1, 0): R2X.e12_split(1, f_); log(f'E12 r1 f{f_}')
for r_ in (3, 2, 1):
    for f_ in (4, 3, 2, 1, 0):
        R2X.e14_split(r_, f_); R2X.e13_split(r_, f_); R2X.e11_split(r_, f_); R2X.e10b_split(r_, f_); RX.e10_split(r_, f_); log(f'E10-E14 r{r_} f{f_}')
for i in (4, 3, 2, 1, 0): RX.e9_split(i); log(f'E9 {i}')
X.e7_temporal(); log('E7')
for cond in reversed(X.E6_CONDITIONS):
    for r_ in (3, 2, 1):
        for f_ in (4, 3, 2, 1, 0): X.e6_job(cond, r_, f_); log(f'E6 {cond} r{r_} f{f_}')
log('helper done')
