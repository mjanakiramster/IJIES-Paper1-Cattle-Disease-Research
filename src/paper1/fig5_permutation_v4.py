"""Figure 5 (Supplementary Figure S6), version 4: permutation null distributions of the FS5 - FS1 ROC-AUC gain with
1,000 permutations per scheme (rev5). Same layout as v3; reads the original 100 permutations and the 900 added ones
(rev5_perm_tables_v1.compute, same aggregation as the table). v3 and its output are not touched.
Output: results/paper1_rev5_v1/figures/Fig5_permutation_null_1000_v1.png/pdf. Run: PYTHONPATH=src python src/paper1/fig5_permutation_v4.py"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper1.rev5_perm_tables_v1 import compute
OUTD = Path(__file__).resolve().parents[2] / 'results' / 'paper1_rev5_v1' / 'figures'; DIS = ['CBPP', 'FMD', 'LSD']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False})
_, M = compute()
obs = M[M.perm == -1].set_index('disease').gain
fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.2), sharex='col')
for i, s in enumerate(('household', 'village')):
    for j, dn in enumerate(DIS):
        ax = axs[i, j]; g = M[(M.scheme == s) & (M.disease == dn) & (M.perm >= 0)].gain; n = len(g); p = (1 + (g >= obs[dn]).sum()) / (1 + n)
        ax.hist(g, bins=30, color='#a9a8a0', edgecolor='white', linewidth=0.3)
        ax.axvline(obs[dn], color='#D55E00', lw=1.6); ax.axvline(0, color='#555', lw=0.6, ls=':')
        ax.set_title(f'({"abcdef"[3 * i + j]}) {dn}, {s} null', loc='left', fontsize=8.5)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
        ax.text(0.03, 0.96, f'observed {obs[dn]:+.3f}, p = {p:.4f}', transform=ax.transAxes, ha='left', va='top', fontsize=7.5, color='#B34700', bbox=dict(fc='white', ec='none', pad=1.0))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if i == 1: ax.set_xlabel('FS5 - FS1 ROC-AUC')
        if j == 0: ax.set_ylabel(f'Permutations (of {n:,})')
fig.tight_layout(h_pad=1.2)
OUTD.mkdir(parents=True, exist_ok=True)
for ext in ('png', 'pdf'): fig.savefig(OUTD / f'Fig5_permutation_null_1000_v1.{ext}', dpi=300, bbox_inches='tight')
print('saved Fig5_permutation_null_1000_v1')
