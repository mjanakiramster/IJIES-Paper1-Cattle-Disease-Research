"""Revision tables and figures for IJIES Paper 1 (no optimisation content). Reads checkpoints from results/paper1_v1/ and
results/paper1_rev_v1/, writes tables (CSV + one Excel workbook) and figures (PNG 300 dpi + PDF) to results/paper1_rev_v1/.
Run from the project root: python src/paper1/rev_analysis_v1.py"""
import sys, json, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper1.core import setup, auc, ecdf_rank, within_year_auc, corrected_t, platt, youden_threshold, load_library, DIS
ROOT = Path(__file__).resolve().parents[2]; P1 = ROOT / 'results' / 'paper1_v1'; REV = ROOT / 'results' / 'paper1_rev_v1'
TAB = REV / 'tables'; FIG = REV / 'figures'; TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
COL = {'FS1': '#7a7a74', 'FS5': '#0072B2', '(a) Rank average': '#D55E00', '(b) Inner-selected single': '#009E73'}
MODELS = list(COL); RATIO = 668 / 2672
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 9.5,
                     'axes.titleweight': 'bold', 'legend.frameon': False, 'savefig.dpi': 300})
df, Y, _, _ = setup(); YR = df.Year.values; FM = pd.read_csv(P1 / 'metrics' / 'fold_metrics_v2.csv')
def save(fig, name): fig.savefig(FIG / f'{name}.png', bbox_inches='tight'); fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight'); plt.close(fig)
def ci(m, lo, hi): return f'{m:+.3f} [{lo:+.3f}, {hi:+.3f}]'
TABLES = {}


# ------------------------------------------------------------------ random vs spatial CV (E9 + development)
def random_cv_metrics():
    rows = []
    for i in range(5):
        z = np.load(REV / 'E9_random_cv' / f'random_{i}.npz'); tr, te = z['train_idx'], z['test_idx']
        for d, dn in enumerate(DIS):
            ytr, yte = Y[tr, d], Y[te, d]; Po, Pt = z['lib_oof'][:, d, :], z['lib_test'][:, d, :]
            Uo = np.column_stack([ecdf_rank(Po[:, m], Po[:, m]) for m in range(23)]).mean(1); Ut = np.column_stack([ecdf_rank(Po[:, m], Pt[:, m]) for m in range(23)]).mean(1)
            jb = int(np.argmax([auc(ytr, Po[:, m]) for m in range(23)]))
            sc = {'FS1': z['fs_test'][:, d, 0], 'FS5': z['fs_test'][:, d, 1], '(a) Rank average': Ut, '(b) Inner-selected single': Pt[:, jb]}
            for m, s in sc.items():
                rows.append({'fold': i, 'disease': dn, 'model': m, 'ROC_AUC': auc(yte, s), 'within_year_AUC': within_year_auc(yte, s, YR[te])[0]})
    return pd.DataFrame(rows)


def table_random_vs_spatial():
    R = random_cv_metrics(); R.to_csv(REV / 'E9_random_cv' / 'random_cv_fold_metrics_v1.csv', index=False); rows = []
    for dn in DIS:
        for m in MODELS:
            r = R[(R.disease == dn) & (R.model == m)]; s = FM[(FM.disease == dn) & (FM.model == m)]
            rows.append({'Disease': dn, 'Model': m, 'Random CV ROC-AUC, mean (SD) of 5 folds': f'{r.ROC_AUC.mean():.3f} ({r.ROC_AUC.std():.3f})',
                         'Spatial CV ROC-AUC, mean (SD) of 15 folds': f'{s.ROC_AUC.mean():.3f} ({s.ROC_AUC.std():.3f})', 'Random minus spatial': f'{r.ROC_AUC.mean() - s.ROC_AUC.mean():+.3f}',
                         'Random CV within-year AUC': f'{r.within_year_AUC.mean():.3f}', 'Spatial CV within-year AUC': f'{s.within_year_AUC.mean():.3f}',
                         '_rand': r.ROC_AUC.mean(), '_spat': s.ROC_AUC.mean(), '_rand_wy': r.within_year_AUC.mean(), '_spat_wy': s.within_year_AUC.mean()})
    T = pd.DataFrame(rows); T.drop(columns=[c for c in T if c.startswith('_')]).to_csv(TAB / 'RT1_random_vs_spatial_cv_v1.csv', index=False)
    TABLES['Random vs spatial CV'] = T.drop(columns=[c for c in T if c.startswith('_')]); return T


def fig_random_vs_spatial(T):
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
    for k, (ax, dn) in enumerate(zip(axs, DIS)):
        s = T[T.Disease == dn]
        for j, (_, r) in enumerate(s.iterrows()):
            c = COL[r.Model]; x = [0, 1]
            ax.plot(x, [r._rand, r._spat], color=c, lw=1.2, marker='o', ms=4.5, label=r.Model if k == 0 else None)
        ax.axhline(0.5, color='#999', lw=0.6, ls=':'); ax.set_xticks([0, 1]); ax.set_xticklabels(['Random\n5-fold CV', 'Spatial block\nCV (81 km)'])
        ax.set_xlim(-0.35, 1.35); ax.set_title(f'({"abc"[k]}) {dn}', loc='left')
        if k == 0: ax.set_ylabel('ROC-AUC (mean over folds)')
    axs[0].set_ylim(0.45, 0.9); fig.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.07), fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, 'Fig3_random_vs_spatial_cv_v1')


# ------------------------------------------------------------------ FS1 vs FS5, paired folds
def fig_fs_paired():
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for k, (ax, dn) in enumerate(zip(axs, DIS)):
        a = FM[(FM.disease == dn) & (FM.model == 'FS1')].sort_values(['rep', 'fold']); b = FM[(FM.disease == dn) & (FM.model == 'FS5')].sort_values(['rep', 'fold'])
        lo, hi = 0.40, 0.85; ax.plot([lo, hi], [lo, hi], color='#999', lw=0.7, ls='--')
        for rep, mk in zip((1, 2, 3), ('o', 's', '^')):
            q = a.rep.values == rep; ax.scatter(a.ROC_AUC.values[q], b.ROC_AUC.values[q], s=18, marker=mk, color='#0072B2', edgecolor='white', linewidth=0.5, label=f'Repeat {rep}' if k == 0 else None)
        m, l, h, p = corrected_t(b.ROC_AUC.values - a.ROC_AUC.values, 1, RATIO, 15)
        ax.text(0.03, 0.97, f'FS5 - FS1 = {m:+.3f}\n95% CI {l:+.3f} to {h:+.3f}\nabove diagonal: {(b.ROC_AUC.values > a.ROC_AUC.values).sum()}/15', transform=ax.transAxes, va='top', fontsize=7.3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal'); ax.set_title(f'({"abc"[k]}) {dn}', loc='left'); ax.set_xlabel('FS1 ROC-AUC')
        if k == 0: ax.set_ylabel('FS5 ROC-AUC')
    fig.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=8); fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, 'Fig4_FS1_vs_FS5_paired_folds_v1')


# ------------------------------------------------------------------ campaign-specific vs pooled models (E10)
def table_campaign_models():
    E = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(REV / 'E10_campaign_cv' / '*.csv')))]); E['neg'] = E.test_households - E.test_positives
    E = E[(E.test_positives >= 5) & (E.neg >= 5)].copy(); E['w'] = E.test_positives * E.neg; rows = []; fold_rows = []
    for dn in DIS:
        for fs in ('FS1', 'FS5'):
            s = E[E.disease == dn]; per = s.groupby(['rep', 'fold']).apply(lambda g: pd.Series({'pooled': np.average(g[f'pooled_{fs}_ROC_AUC'], weights=g.w), 'campaign': np.average(g[f'campaign_{fs}_ROC_AUC'], weights=g.w)}), include_groups=False)
            m, l, h, p = corrected_t(per.campaign.values - per.pooled.values, 1, RATIO, 15)
            rows.append({'Disease': dn, 'Features': fs, 'Fold x year units': len(s), 'Pooled-campaign model, within-campaign AUC': f'{per.pooled.mean():.3f}',
                         'Campaign-specific model, within-campaign AUC': f'{per.campaign.mean():.3f}', 'Campaign-specific minus pooled [corrected-t 95% CI]': ci(m, l, h),
                         'Folds favouring campaign-specific': f'{int((per.campaign > per.pooled).sum())}/15', '_m': m, '_pool': per.pooled.mean(), '_camp': per.campaign.mean(), '_lo': l, '_hi': h})
            for (r, f), v in per.iterrows(): fold_rows.append({'disease': dn, 'features': fs, 'rep': r, 'fold': f, **v.to_dict()})
    T = pd.DataFrame(rows); T.drop(columns=[c for c in T if c.startswith('_')]).to_csv(TAB / 'RT2_campaign_specific_vs_pooled_v1.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(REV / 'E10_campaign_cv' / 'campaign_vs_pooled_fold_level_v1.csv', index=False)
    TABLES['Campaign vs pooled models'] = T.drop(columns=[c for c in T if c.startswith('_')]); return T


def fig_campaign(T):
    YA = pd.read_csv(P1 / 'tables' / 'TableS_year_specific_AUC_v1.csv'); D5 = pd.read_csv(REV / 'design_checks' / 'D5_campaign_descriptors_v1.csv')
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={'width_ratios': [1.05, 1]})
    ax = axs[0]; yrs = D5['Survey year'].values; x = np.arange(len(yrs)); w = 0.26; pal = {'CBPP': '#7a7a74', 'FMD': '#0072B2', 'LSD': '#D55E00'}
    for j, dn in enumerate(DIS):
        ax.bar(x + (j - 1) * w, D5[f'{dn} prevalence (%)'], w, color=pal[dn], label=dn)
    ax.set_xticks(x); ax.set_xticklabels([f'{y}\n(n={n})' for y, n in zip(yrs, D5.Households)], fontsize=7.5); ax.set_ylabel('Household-reported prevalence (%)')
    ax.set_title('(a) Prevalence by survey campaign', loc='left'); ax.legend(fontsize=7.5, ncol=3, loc='upper left')
    ax = axs[1]; ypos = np.arange(len(DIS))
    for j, fs in enumerate(('FS1', 'FS5')):
        s = T[T.Features == fs].set_index('Disease').reindex(DIS); off = (j - 0.5) * 0.32
        ax.errorbar(s._m, ypos + off, xerr=[s._m - s._lo, s._hi - s._m], fmt='o', color=COL[fs], ms=5, capsize=2, lw=1, label=fs)
    ax.axvline(0, color='#555', lw=0.6); ax.set_yticks(ypos); ax.set_yticklabels(DIS); ax.invert_yaxis(); ax.set_xlabel('Within-campaign ROC-AUC difference\n(campaign-specific minus pooled model)')
    ax.set_title('(b) Training on one campaign vs all', loc='left'); ax.legend(fontsize=7.5, loc='lower right')
    fig.tight_layout(); save(fig, 'Fig5_campaign_prevalence_and_models_v1')


# ------------------------------------------------------------------ forward temporal validation figure
def fig_temporal():
    T12 = pd.read_csv(P1 / 'tables' / 'Table12_forward_temporal_validation_v1.csv')
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.7), gridspec_kw={'width_ratios': [1, 1.1]})
    ax = axs[0]; x = np.arange(3); w = 0.36; tr = [100 * float(T12[T12.Disease == d]['Train prevalence'].iloc[0]) for d in DIS]; ts = [100 * float(T12[T12.Disease == d]['Test prevalence'].iloc[0]) for d in DIS]
    ax.bar(x - w / 2, tr, w, color='#b9b9b2', label='Train 2012-2015'); ax.bar(x + w / 2, ts, w, color='#0072B2', label='Test 2018-2019')
    ax.set_xticks(x); ax.set_xticklabels(DIS); ax.set_ylabel('Prevalence (%)'); ax.set_title('(a) Prevalence shift', loc='left'); ax.legend(fontsize=7.5)
    ax = axs[1]
    for j, fs in enumerate(('FS1', 'FS5')):
        s = T12[T12.Features == fs].set_index('Disease').reindex(DIS); v = s['ROC-AUC [95% CI]'].str.replace('[', ' ').str.replace(']', ' ').str.replace(',', ' ').str.split()
        m = np.array([float(a[0]) for a in v]); lo = np.array([float(a[1]) for a in v]); hi = np.array([float(a[2]) for a in v]); off = (j - 0.5) * 0.3
        ax.errorbar(m, np.arange(3) + off, xerr=[m - lo, hi - m], fmt='o', color=COL[fs], ms=5, capsize=2, lw=1, label=fs)
    ax.axvline(0.5, color='#555', lw=0.6, ls=':'); ax.set_yticks(range(3)); ax.set_yticklabels(DIS); ax.invert_yaxis(); ax.set_xlabel('ROC-AUC on 2018-2019 (95% CI)')
    ax.set_title('(b) Forward temporal ROC-AUC', loc='left'); ax.legend(fontsize=7.5, loc='lower right'); fig.tight_layout(); save(fig, 'Fig8_forward_temporal_v1')


# ------------------------------------------------------------------ 23-model library comparison (corrected development results)
def table_library():
    G = load_library(); names = [m[0] for m in G['LIB']] + ['LightGBM [ELK]', 'SVM-RBF [ELK]']; rows = []
    for i in range(5, 20):
        z = np.load(P1 / 'E3_library23' / f'split_{i:02d}.npz'); te = z['test_idx']
        for d, dn in enumerate(DIS):
            for m in range(23): rows.append({'split': i, 'disease': dn, 'model': names[m], 'ROC_AUC': auc(Y[te, d], z['test'][:, d, m])})
    L = pd.DataFrame(rows); S = L.groupby(['model', 'disease']).ROC_AUC.agg(['mean', 'std']).unstack('disease')
    out = pd.DataFrame({'Model [feature set]': S.index})
    for dn in DIS: out[f'{dn} ROC-AUC mean (SD)'] = [f'{a:.3f} ({b:.3f})' for a, b in zip(S[('mean', dn)], S[('std', dn)])]
    out['_order'] = [names.index(n) for n in S.index]; out = out.sort_values('_order').drop(columns='_order')
    out.to_csv(TAB / 'RT3_library_models_spatial_v1.csv', index=False); TABLES['23-model library'] = out; L.to_csv(REV / 'library23_fold_auc_v1.csv', index=False); return out


def excel_workbook():
    import shutil
    TB = P1 / 'tables'
    main = {'T1 Dataset': 'Table1_dataset_prevalence_v1', 'T2 Feature sets': 'Table2_feature_sets_v1', 'T4 Spatial performance': 'Table4_overall_spatial_performance_v1',
            'T4b Within-year': 'Table5_within_year_performance_v1', 'T5 Paired differences': 'Table7_disease_specific_comparisons_v1', 'T7 Proximity': 'Table9_proximity_removal_v1',
            'T7b Block size': 'Table8_block_size_sensitivity_v1', 'T8 Negative controls': 'Table10_negative_controls_v1', 'T9 Temporal': 'Table12_forward_temporal_validation_v1',
            'S Year-specific AUC': 'TableS_year_specific_AUC_v1', 'S Code corrections': 'Table11_ensemble_prior_sensitivity_v1', 'S Grouped importance': 'TableS_grouped_permutation_importance_v1',
            'S Confirmatory': 'TableS_confirmatory_as_observed_confirmatory_summary_v1', 'S Optimisation (as run)': 'TableS_optimisation_stage5_as_run_v1', 'S Feature dictionary': 'feature_dictionary_v1'}
    with pd.ExcelWriter(TAB / 'IJIES_Paper1_revised_tables_v1.xlsx') as W:
        pd.read_csv(TB / 'Table3_models_v2.csv').to_excel(W, sheet_name='T3 Models', index=False)
        for sh, f in main.items(): pd.read_csv(TB / f'{f}.csv').to_excel(W, sheet_name=sh[:31], index=False)
        for sh, t in TABLES.items(): t.to_excel(W, sheet_name=('R ' + sh)[:31], index=False)
        for f, sh in (('D4_variogram_ranges_v1', 'R Variograms'), ('D5_campaign_descriptors_v1', 'R Campaigns'), ('D5_variance_shares_v1', 'R Variance shares'), ('D1_fold_structure_v1', 'R Fold structure')):
            pd.read_csv(REV / 'design_checks' / f'{f}.csv').to_excel(W, sheet_name=sh, index=False)


if __name__ == '__main__':
    T = table_random_vs_spatial(); fig_random_vs_spatial(T); print(TABLES['Random vs spatial CV'].to_string())
    fig_fs_paired(); C = table_campaign_models(); fig_campaign(C); print(TABLES['Campaign vs pooled models'].to_string())
    fig_temporal(); print(table_library().to_string()); excel_workbook(); print('done')
