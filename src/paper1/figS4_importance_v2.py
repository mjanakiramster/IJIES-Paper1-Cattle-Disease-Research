"""Supplementary Figure S4, version 2: grouped permutation importance (FS5 ExtraTrees).
v2 changes from v1: wider panels, at most four x ticks so the tick labels do not run together, and a shared x range.
Reads results/paper1_v1/tables/TableS_grouped_permutation_importance_v1.csv. Run: python src/paper1/figS4_importance_v2.py"""
from pathlib import Path
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
P1 = Path(__file__).resolve().parents[2] / 'results' / 'paper1_v1'; DIS = ['CBPP', 'FMD', 'LSD']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False})
I = pd.read_csv(P1 / 'tables' / 'TableS_grouped_permutation_importance_v1.csv')
groups = ['Own-disease rates (6)', 'Cross-disease rates (12)', 'Sampling density (6)', 'Coordinates (4)', 'Climate indices (6)', 'Environmental (5)']
lo = (I.mean_AUC_drop - I.sd_over_folds).min(); hi = (I.mean_AUC_drop + I.sd_over_folds).max(); pad = 0.1 * (hi - lo)
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.9), sharey=True)
for k, (ax, dn) in enumerate(zip(axs, DIS)):
    g = I[I.disease == dn].set_index('group').reindex(groups)
    ax.barh(range(len(groups)), g.mean_AUC_drop, xerr=g.sd_over_folds, color='#0072B2', height=0.6, error_kw={'lw': 0.7, 'capsize': 2})
    ax.axvline(0, color='#555', lw=0.6); ax.set_xlim(lo - pad, hi + pad); ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups, fontsize=7.5); ax.set_title(f'({"abc"[k]}) {dn}', loc='left')
    ax.set_xlabel('Drop in ROC-AUC')
fig.text(0.99, -0.02, 'Bars: mean over 15 development folds; whiskers: SD over folds', ha='right', fontsize=7, color='#555')
fig.tight_layout()
for ext in ('png', 'pdf'): fig.savefig(P1 / 'figures' / f'FigS4_grouped_importance_v2.{ext}', dpi=300, bbox_inches='tight')
print('saved FigS4_grouped_importance_v2')
