"""Revised figures, version 3: Fig6 v3 moves the legend below the axis and the column header next to the rows (v2 legend covered CBPP).
Earlier: revised figures, version 2 (replace two figures from rev_analysis_v1.py; v1 files are kept).
Fig4_FS1_vs_FS5_paired_folds_v2: annotation moved to the empty lower-right corner with a white box (v1 overlapped FMD points).
Fig6_campaign_specific_vs_pooled_v2: only the model comparison (v1 panel a repeated Figure 2b); legend no longer covers the LSD intervals.
Run from the project root: python src/paper1/rev_figures_v2.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runpy, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
G = runpy.run_path(str(Path(__file__).with_name('rev_analysis_v1.py')), run_name='lib')
FM, DIS, COL, corrected_t, RATIO, save = G['FM'], G['DIS'], G['COL'], G['corrected_t'], G['RATIO'], G['save']


def fig_fs_paired_v2():
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.75))
    for k, (ax, dn) in enumerate(zip(axs, DIS)):
        a = FM[(FM.disease == dn) & (FM.model == 'FS1')].sort_values(['rep', 'fold']); b = FM[(FM.disease == dn) & (FM.model == 'FS5')].sort_values(['rep', 'fold'])
        lo, hi = 0.40, 0.85; ax.plot([lo, hi], [lo, hi], color='#999', lw=0.7, ls='--')
        for rep, mk in zip((1, 2, 3), ('o', 's', '^')):
            q = a.rep.values == rep
            ax.scatter(a.ROC_AUC.values[q], b.ROC_AUC.values[q], s=18, marker=mk, color='#0072B2', edgecolor='white', linewidth=0.5, label=f'Repeat {rep}' if k == 0 else None, zorder=3)
        m, l, h, _ = corrected_t(b.ROC_AUC.values - a.ROC_AUC.values, 1, RATIO, 15)
        ax.text(0.97, 0.04, f'FS5 - FS1 = {m:+.3f}\n95% CI {l:+.3f} to {h:+.3f}\nFS5 higher in {(b.ROC_AUC.values > a.ROC_AUC.values).sum()}/15 folds', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=7, bbox=dict(fc='white', ec='none', alpha=0.9, pad=1.5), zorder=4)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal'); ax.set_title(f'({"abc"[k]}) {dn}', loc='left'); ax.set_xlabel('FS1 ROC-AUC')
        if k == 0: ax.set_ylabel('FS5 ROC-AUC')
    fig.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=8); fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, 'Fig4_FS1_vs_FS5_paired_folds_v2')


def fig_campaign_models_v3():
    T = G['table_campaign_models'](); fig, ax = plt.subplots(figsize=(5.4, 2.8)); ypos = np.arange(len(DIS))
    for j, fs in enumerate(('FS1', 'FS5')):
        s = T[T.Features == fs].set_index('Disease').reindex(DIS); off = (j - 0.5) * 0.3
        ax.errorbar(s._m, ypos + off, xerr=[s._m - s._lo, s._hi - s._m], fmt='o', color=COL[fs], ms=5, capsize=2, lw=1, label=fs)
        for yy, (_, r) in zip(ypos + off, s.iterrows()):
            ax.text(0.205, yy, f'{r._pool:.3f} \u2192 {r._camp:.3f}', va='center', fontsize=7, color=COL[fs])
    ax.axvline(0, color='#555', lw=0.6); ax.set_yticks(ypos); ax.set_yticklabels(DIS); ax.set_ylim(2.45, -0.75); ax.set_xlim(-0.12, 0.30)
    ax.text(0.205, -0.55, 'pooled \u2192 campaign-specific', fontsize=7, color='#555', va='center')
    ax.set_xlabel('Within-campaign ROC-AUC: campaign-specific minus pooled model')
    ax.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2); fig.tight_layout(); save(fig, 'Fig6_campaign_specific_vs_pooled_v3')


if __name__ == '__main__':
    fig_campaign_models_v3(); print('saved Fig6 v3')
