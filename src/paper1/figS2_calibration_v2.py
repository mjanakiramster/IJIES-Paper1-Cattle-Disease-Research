"""Supplementary Figure S2, version 2: calibration by deciles of predicted probability (pooled development predictions, repeat 1).
v2 changes from v1: shorter x-axis label so the panels' labels no longer overlap, panel letters, and wider spacing.
Run from the project root: python src/paper1/figS2_calibration_v2.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper1 import analysis as A
from paper1.core import setup, DIS
P = A.collect_predictions(); df, Y, _, _ = setup(); fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7))
for k, (ax, dn) in enumerate(zip(axs, DIS)):
    d = DIS.index(dn)
    for m in ('FS5', '(a) Rank average'):
        idx, p = A.pooled(P, m, dn, 1, 'prob'); y = Y[idx, d]; bins = np.unique(np.quantile(p, np.linspace(0, 1, 11))); kk = np.clip(np.digitize(p, bins[1:-1]), 0, len(bins) - 2)
        ax.plot([p[kk == j].mean() for j in range(len(bins) - 1)], [y[kk == j].mean() for j in range(len(bins) - 1)], 'o-', color=A.COL[m], ms=3, lw=1, label=m)
    ax.plot([0, 1], [0, 1], ':', color='#999', lw=0.8); ax.set_xlim(0, 0.75); ax.set_ylim(0, 0.75); ax.set_title(f'({"abc"[k]}) {dn}', loc='left'); ax.set_xlabel('Mean predicted probability')
axs[0].set_ylabel('Observed proportion'); axs[0].legend(fontsize=7, loc='upper left'); fig.tight_layout(w_pad=1.5)
fig.text(0.99, -0.03, 'Points: deciles of predicted probability', ha='right', fontsize=7, color='#555'); A.save_fig(fig, 'FigS2_calibration_v2'); print('saved FigS2_calibration_v2')
