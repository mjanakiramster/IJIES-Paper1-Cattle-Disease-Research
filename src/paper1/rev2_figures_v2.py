"""Revision-2 figures, version 2: layout fixes only, from the saved v1 tables (no new computation of results).
R2Fig1_ablation_v2: legend of panel (b) moved below the panel (it touched an error bar in v1).
R2Fig6_train_test_distances_v2: legend moved to the empty left part of the plot (it overlapped the curves in v1).
Run from the project root: python src/paper1/rev2_figures_v2.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]; R2 = ROOT / 'results/paper1_rev2_v1'; TAB, FIG = R2 / 'tables', R2 / 'figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold', 'legend.frameon': False, 'savefig.dpi': 300})
DIS = ['CBPP', 'FMD', 'LSD']; SETS = ['FS1', 'FS2', 'FS3', 'FS4', 'FS5']; COL = ('#7a7a74', '#0072B2', '#D55E00')
def save(fig, name): fig.savefig(FIG / f'{name}.png', bbox_inches='tight'); fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight'); plt.close(fig)
def parse(s):  # '+0.040 [-0.015, +0.095]' -> (m, lo, hi)
    a = s.replace('[', ' ').replace(']', ' ').replace(',', ' ').split(); return float(a[0]), float(a[1]), float(a[2])


def fig_ablation():
    T = pd.read_csv(TAB / 'R2T1_ablation_performance_v1.csv'); I = pd.read_csv(TAB / 'R2T2_ablation_increments_v1.csv')
    T['_roc'] = T['ROC-AUC fold mean (SD)'].str.split(' ').str[0].astype(float); T['_wy'] = T['Within-year AUC fold mean'].astype(float)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={'width_ratios': [1, 1.25]})
    ax = axs[0]
    for dn, mk, c in zip(DIS, ('o', 's', '^'), COL):
        s = T[T.Disease == dn]; ax.plot(range(5), s._roc, marker=mk, color=c, lw=1.2, ms=4.5, label=f'{dn} overall'); ax.plot(range(5), s._wy, marker=mk, color=c, lw=1, ls='--', mfc='white', ms=4.5, label=f'{dn} within-year')
    ax.set_xticks(range(5)); ax.set_xticklabels(SETS); ax.axhline(0.5, color='#999', lw=0.6, ls=':'); ax.set_ylabel('ROC-AUC (mean of 15 folds)'); ax.set_title('(a) Progressive feature sets', loc='left')
    ax.legend(fontsize=6.5, ncol=2, loc='upper left', bbox_to_anchor=(0, -0.16))
    ax = axs[1]; steps = list(dict.fromkeys(I.Step)); yb = np.arange(len(steps))
    for j, (dn, c) in enumerate(zip(DIS, COL)):
        s = I[I.Disease == dn].set_index('Step').reindex(steps); v = np.array([parse(x) for x in s['Fold mean [corrected-t 95% CI]']]); off = (j - 1) * 0.25
        ax.errorbar(v[:, 0], yb + off, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt='o', color=c, ms=4, capsize=2, lw=1, label=dn)
    ax.axvline(0, color='#555', lw=0.6); ax.set_yticks(yb); ax.set_yticklabels([f'{s}\n+{I[I.Step == s]["Added group"].iloc[0].split(" (")[0].lower()}' for s in steps], fontsize=7); ax.invert_yaxis()
    ax.set_xlabel('ROC-AUC change (corrected-t 95% CI)'); ax.set_title('(b) Contribution of each added group', loc='left')
    ax.legend(fontsize=7, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout(); save(fig, 'R2Fig1_ablation_v2')


def fig_distances():
    from paper1.core import setup
    from paper1.experiments import _blocks
    df, Y, NB, FB = setup(); LOC = df['loc'].values; D = NB.D; rows = []
    def add(scheme, te, tr):
        tl = np.unique(LOC[tr]); rows.extend({'scheme': scheme, 'distance_km': v} for v in D[np.ix_(LOC[te], tl)].min(1))
    for f in range(5): add('Random 5-fold CV', np.where(df.random_fold.values == f)[0], np.where(df.random_fold.values != f)[0])
    for r in (1, 2, 3):
        fo = df[f'spatial_fold_r{r}'].values
        for f in range(5):
            te = np.where(fo == f)[0]; tr = np.where(fo != f)[0]; add('Spatial 81 km', te, tr); dmin = D[:, np.unique(LOC[te])].min(1)
            for bkm in (50, 100): add(f'Spatial 81 km + {bkm} km buffer', te, tr[dmin[LOC[tr]] > bkm])
    for bkm in (120.0, 160.0):
        bl = _blocks(df, bkm)
        for r in (1, 2, 3):
            fo = bl[r][1]
            for f in range(5): add(f'Spatial {int(bkm)} km', np.where(fo == f)[0], np.where(fo != f)[0])
    Dd = pd.DataFrame(rows); order = ['Random 5-fold CV', 'Spatial 81 km', 'Spatial 120 km', 'Spatial 160 km', 'Spatial 81 km + 50 km buffer', 'Spatial 81 km + 100 km buffer']
    chk = Dd.groupby('scheme').distance_km.median().reindex(order).round(1).values; ref = pd.read_csv(TAB / 'R2T9_train_test_distances_v1.csv')['Median (km)'].values
    assert np.allclose(chk, ref), (chk, ref)   # same data as the v1 table
    fig, ax = plt.subplots(figsize=(5.6, 3.0)); cols = ['#D55E00', '#0072B2', '#4292c6', '#9ecae1', '#7a7a74', '#b9b9b2']; lss = ['-', '-', '--', ':', '-.', '-.']
    for s, c, ls in zip(order, cols, lss):
        v = np.sort(Dd[Dd.scheme == s].distance_km.values); ax.plot(np.maximum(v, 0.5), np.arange(1, len(v) + 1) / len(v), color=c, ls=ls, lw=1.4, label=s)
    ax.set_xscale('log'); ax.set_xlabel('Distance from test household to nearest training location (km, log scale; 0 shown at 0.5)'); ax.set_ylabel('Cumulative share of test households')
    ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(0.08, 0.45)); fig.tight_layout(); save(fig, 'R2Fig6_train_test_distances_v2')


if __name__ == '__main__':
    fig_ablation(); fig_distances(); print('saved R2Fig1_ablation_v2, R2Fig6_train_test_distances_v2')
