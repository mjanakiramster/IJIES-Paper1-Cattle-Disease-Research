"""Figure 5, version 3 (v3 moves the observed-value label to the top-left with headroom so it no longer overlaps the line or bars).
Figure 5, version 2: permutation null distributions of the FS5 - FS1 ROC-AUC gain.
v2 changes from v1: integer count axis, shorter panel titles that do not run into each other, and the permutation p-value in each panel.
Reads results/paper1_v1/E5_permutations/*.csv (same aggregation as analysis.table10). Run: python src/paper1/fig5_permutation_v3.py"""
from pathlib import Path
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
P1 = Path(__file__).resolve().parents[2] / 'results' / 'paper1_v1'; DIS = ['CBPP', 'FMD', 'LSD']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False})
A = pd.concat([pd.read_csv(p) for p in sorted((P1 / 'E5_permutations').glob('*.csv'))], ignore_index=True)
M = A.groupby(['scheme', 'perm', 'disease'])[['FS1_ROC_AUC', 'FS5_ROC_AUC']].mean().reset_index(); M['gain'] = M.FS5_ROC_AUC - M.FS1_ROC_AUC
obs = M[M.perm == -1].set_index('disease').gain
fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.2), sharex='col')
for i, s in enumerate(('household', 'village')):
    for j, dn in enumerate(DIS):
        ax = axs[i, j]; g = M[(M.scheme == s) & (M.disease == dn) & (M.perm >= 0)].gain; p = (1 + (g >= obs[dn]).sum()) / (1 + len(g))
        ax.hist(g, bins=20, color='#a9a8a0', edgecolor='white', linewidth=0.3)
        ax.axvline(obs[dn], color='#D55E00', lw=1.6); ax.axvline(0, color='#555', lw=0.6, ls=':')
        ax.set_title(f'({"abcdef"[3 * i + j]}) {dn}, {s} null', loc='left', fontsize=8.5)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
        ax.text(0.03, 0.96, f'observed {obs[dn]:+.3f}, p = {p:.3f}', transform=ax.transAxes, ha='left', va='top', fontsize=7.5, color='#B34700', bbox=dict(fc='white', ec='none', pad=1.0))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if i == 1: ax.set_xlabel('FS5 - FS1 ROC-AUC')
        if j == 0: ax.set_ylabel('Permutations (of 100)')
fig.tight_layout(h_pad=1.2)
for ext in ('png', 'pdf'): fig.savefig(P1 / 'figures' / f'Fig5_permutation_null_v3.{ext}', dpi=300, bbox_inches='tight')
print('saved Fig5_permutation_null_v3')
