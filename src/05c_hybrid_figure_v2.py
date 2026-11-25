"""Figure for Stage 5c-A, v2 (fixes v1: bars and value labels were misaligned because the two groups were drawn
in separate calls). Mean rank of all 23 optimisers on the shifted benchmark problems (lower = better)."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / 'results/stage5c_hybrids_v1'
R = pd.read_csv(OUT / 'benchmark_ranks_all.csv').sort_values('mean_rank_shifted', ascending=False).reset_index(drop=True)
plt.rcParams.update({'font.size': 9.5, 'axes.spines.top': False, 'axes.spines.right': False})
col = {'hybrid': '#2a78d6', 'base': '#a9a8a0'}
fig, ax = plt.subplots(figsize=(7.2, 7.4)); y = np.arange(len(R))
ax.barh(y, R.mean_rank_shifted, color=[col[k] for k in R.kind], height=0.68)
for yi, v in zip(y, R.mean_rank_shifted): ax.text(v + 0.2, yi, f'{v:.1f}', va='center', fontsize=8, color='#444')
ax.set_yticks(y); ax.set_yticklabels(R.optimiser); ax.set_xlim(0, R.mean_rank_shifted.max() + 2)
ax.set_xlabel('Mean rank on shifted benchmark problems\n(1 = best of 23; 4 functions x D = 10, 30 x 5 seeds)')
ax.set_title('Benchmark check of the 15 pairwise hybrids (optimum away from the centre)', loc='left', fontsize=10)
ax.legend(handles=[Patch(color=col['hybrid'], label='Pairwise relay hybrid'), Patch(color=col['base'], label='Base optimiser')], frameon=False, loc='lower right')
ax.grid(axis='x', color='#eeeeea'); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT / 'figures' / 'hybrid_benchmark_ranks_v2.png', dpi=200, bbox_inches='tight')
