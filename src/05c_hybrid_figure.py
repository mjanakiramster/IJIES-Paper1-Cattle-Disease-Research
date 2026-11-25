"""Figure for Stage 5c-A: mean rank of all 23 optimisers on the shifted benchmark problems (lower = better)."""
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / 'results/stage5c_hybrids_v1'
R = pd.read_csv(OUT / 'benchmark_ranks_all.csv').sort_values('mean_rank_shifted', ascending=False)
plt.rcParams.update({'font.size': 9.5, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(7.2, 7.4))
col = {'hybrid': '#2a78d6', 'base': '#a9a8a0'}
for k in ('base', 'hybrid'):
    s = R[R.kind == k]; ax.barh(s.optimiser, s.mean_rank_shifted, color=col[k], height=0.68, label='Pairwise relay hybrid' if k == 'hybrid' else 'Base optimiser')
for y, (o, v) in enumerate(zip(R.optimiser, R.mean_rank_shifted)): ax.text(v + 0.2, y, f'{v:.1f}', va='center', fontsize=8, color='#444')
ax.set_xlabel('Mean rank on shifted benchmark problems (1 = best of 23; 4 functions x D = 10, 30 x 5 seeds)')
ax.set_title('Benchmark check of the 15 pairwise hybrids (optimum away from the centre)', loc='left', fontsize=10)
ax.legend(frameon=False, loc='lower right'); ax.grid(axis='x', color='#eeeeea'); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT / 'figures' / 'hybrid_benchmark_ranks_v1.png', dpi=200, bbox_inches='tight')
