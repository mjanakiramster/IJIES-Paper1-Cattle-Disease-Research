"""Builds the revised IJIES Paper 1 notebook (24 sections, no optimisation content in the main analysis).
Writes IJIES_Paper1_Revised_Experiments_unexecuted_v1.ipynb in the project root. Run from the project root:
python src/paper1/build_notebook_rev_v1.py"""
import nbformat as nbf
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Spatial generalisation and survey-campaign confounding in machine-learning models for retrospective mapping of cattle disease occurrence in Ethiopia
**Executable notebook for the revised IJIES manuscript.** It reproduces every table and figure from the raw data, in 24 sections.

**How it runs.** The heavy experiments are in `src/paper1/` and write checkpoints under `results/`. When a checkpoint exists it is loaded; delete `results/paper1_v1/` and `results/paper1_rev_v1/` to recompute everything (about 4 hours on 2 CPU cores). With checkpoints the notebook runs in a few minutes.

**Data.** ILRI MELDATA dataset (hdl:20.500.11766.1/FK2/8YFO3D, CC BY 4.0). Place `cbpp.xlsx`, `fmd.xlsx` and `lsd.xlsx` in `data/raw/`. The fold file `data/raw/stage3_folds_v1.csv` is included in the package.

**Scope.** Retrospective mapping of *household-reported* occurrence. The models are not forecasting tools, and the outcomes are not laboratory-confirmed. `Weight_*`, `Year`, `Region`, `Year_region` and identifiers are never predictors.

**Google Colab.** Upload `IJIES_Paper1_code_and_results.zip` and the three data files, then run the first cell. It unzips the package, installs the pinned versions and moves into the project folder.
""")
code("""
# Colab / local setup
import sys, os, subprocess, zipfile
from pathlib import Path
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    if Path('IJIES_Paper1_code_and_results.zip').exists() and not Path('IJIES_Paper1_code_and_results').exists():
        zipfile.ZipFile('IJIES_Paper1_code_and_results.zip').extractall('.')
    os.chdir('IJIES_Paper1_code_and_results')
    for f in ('cbpp.xlsx', 'fmd.xlsx', 'lsd.xlsx'):
        if Path('..', f).exists(): os.replace(Path('..', f), Path('data/raw', f))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / 'src')); os.environ['PYTHONPATH'] = str(ROOT / 'src'); os.chdir(ROOT)
import json, platform, importlib, runpy, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display
warnings.filterwarnings('ignore'); pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 60)
vers = {'python': platform.python_version(), **{m: importlib.import_module(m).__version__ for m in ['numpy', 'pandas', 'scipy', 'sklearn', 'xgboost', 'lightgbm', 'catboost', 'matplotlib']}}
print(json.dumps(vers, indent=1))
import yaml; CFG = yaml.safe_load(open('configs/ijies_paper1_config_v1.yaml')); print('seeds:', CFG.get('seeds'))
P1, REV = ROOT / 'results/paper1_v1', ROOT / 'results/paper1_rev_v1'
def show(name, folder=None, width=900):
    for f in ([folder] if folder else [REV / 'figures', P1 / 'figures']):
        p = Path(f) / f'{name}.png'
        if p.exists(): display(Image(filename=str(p), width=width)); return
    print('missing figure', name)
""")

md("## 1. Data loading and inspection\nThe three files share household IDs and locations. `setup()` merges them, converts coordinates to km, and attaches the fixed fold file.")
code("""
from paper1.core import setup, dev_splits, DIS, FEATURE_SETS, auc
from common.data import AUDITED_MD5, file_hashes, BLOCK_KM
df, Y, NB, FB = setup()
md5 = {f: file_hashes(ROOT / 'data/raw' / f)['md5'] for f in AUDITED_MD5}; assert md5 == AUDITED_MD5
print('raw-file MD5 match the audited values')
print('households', len(df), '| locations', df['loc'].nunique(), '| survey years', sorted(df.Year.unique()))
df[['ID', 'Location_id', 'Year', 'Region', 'Latitude', 'Longitude', 'Altitude', 'Temperature', 'Precipitation', 'Relative_humidity', 'Wind_speed'] + DIS].head()
""")
md("## 2. Data-quality checks")
code("""
pred = ['Altitude', 'Temperature', 'Precipitation', 'Relative_humidity', 'Wind_speed', 'Latitude', 'Longitude']
checks = {'duplicate household IDs': int(df.ID.duplicated().sum()), 'duplicate rows (predictors + outcomes)': int(df.duplicated(pred + DIS).sum()),
          'locations with >1 coordinate pair': int((df.groupby('loc')[['Latitude', 'Longitude']].nunique() > 1).any(axis=1).sum()),
          'locations surveyed in >1 year': int((df.groupby('loc').Year.nunique() > 1).sum()),
          'outcome values other than 0/1': int((~df[DIS].isin([0, 1])).sum().sum()),
          '2012 households labelled Afar but south of 8 deg N (flag only)': int(((df.Year == 2012) & (df.Region == 'Afar') & (df.Latitude < 8)).sum())}
pd.Series(checks, name='count').to_frame()
""")
md("## 3. Missing-value analysis")
code("""
cols = pred + DIS + ['Year', 'Region', 'Location_id']
pd.DataFrame({'missing': df[cols].isna().sum(), 'share (%)': (100 * df[cols].isna().mean()).round(2)})
""")
md("## 4. Target-variable analysis\nReported prevalence by survey year (Table 1) and the overlap between diseases.")
code("""
display(pd.read_csv(P1 / 'tables/Table1_dataset_prevalence_v1.csv'))
co = pd.DataFrame({a: [np.corrcoef(Y[:, i], Y[:, j])[0, 1] for j in range(3)] for i, a in enumerate(DIS)}, index=DIS)
print('household-level phi correlation between diseases'); co.round(3)
""")
md("## 5. Descriptive statistics")
code("""
desc = df[pred[:5]].describe().T.round(2); desc
""")
md("## 6. Exploratory data analysis\nEnvironmental variables by outcome (standardised mean difference between positive and negative households).")
code("""
smd = pd.DataFrame({dn: [(df.loc[Y[:, d] == 1, v].mean() - df.loc[Y[:, d] == 0, v].mean()) / df[v].std() for v in pred[:5]] for d, dn in enumerate(DIS)}, index=pred[:5])
fig, ax = plt.subplots(figsize=(6, 2.6)); smd.plot.barh(ax=ax, color=['#7a7a74', '#0072B2', '#D55E00'], width=0.75); ax.axvline(0, color='k', lw=0.6)
ax.set_xlabel('standardised mean difference (positive - negative)'); plt.tight_layout(); plt.show(); smd.round(3)
""")
md("## 7. Geographical distribution analysis\nSurvey locations by campaign, and campaign descriptors: area, overlap with other campaigns and the spread of location-level prevalence within each campaign.")
code("""
show('Fig2_survey_years_v2')
runpy.run_path('src/paper1/rev_design_checks_v1.py', run_name='__main__')   # writes D1-D5 (folds, inner folds, isolation, variograms, campaigns)
display(pd.read_csv(REV / 'design_checks/D5_campaign_descriptors_v1.csv'))
display(pd.read_csv(REV / 'design_checks/D5_variance_shares_v1.csv').round(3))
""")
md("## 8. Feature engineering\nFS1 = 5 environmental variables. FS5 = 39 features: 5 environmental, 6 climate indices, 4 coordinate terms, 9 kernel disease rates (3 bandwidths x 3 diseases), 9 k-nearest-location rates (3 k values x 3 diseases) and 6 sampling-density terms. Rates use training rows only, with the query location's own 81 km block excluded.")
code("""
r, f, tr, te, bc = dev_splits(df)[0]
_, _, names = FB.build(FEATURE_SETS['FS5'], tr, te, df[bc].values, 1)
fd = pd.read_csv(P1 / 'tables/feature_dictionary_v1.csv') if (P1 / 'tables/feature_dictionary_v1.csv').exists() else pd.DataFrame({'feature': names})
print(len(names), 'FS5 features'); display(pd.read_csv(P1 / 'tables/Table2_feature_sets_v1.csv')); fd
""")
md("## 9. Feature-leakage prevention\nThree checks: (i) flipping all outer-test labels leaves features, inner predictions, calibrated probabilities and thresholds unchanged; (ii) features without outcome labels do not change when labels are permuted; (iii) the corrected k-nearest tie rule matches an independent loop-based implementation.")
code("""
s = json.load(open(REV / 'design_checks/design_checks_summary_v1.json')); print('test-label flip (repeat 1, fold 0, FS5):'); display(pd.DataFrame(s['D3']))
from paper1.experiments import permute_labels
Yp = permute_labels(Y, df, 'household', np.random.default_rng(1)); _, _, _, FBp = setup(Y_override=Yp)
A, _, nm = FB.build(FEATURE_SETS['FS5'], tr, te, df[bc].values, 1); Ap, _, _ = FBp.build(FEATURE_SETS['FS5'], tr, te, df[bc].values, 1)
free = [i for i, n in enumerate(nm) if not n.startswith(('kern', 'knn')) or n.startswith('knndist')]
print('label-free columns unchanged under permutation:', bool(np.allclose(A[:, free], Ap[:, free])), '| label-based columns changed:', not np.allclose(A, Ap))
print(json.load(open(P1 / 'verification/P4_tiebreak_verification_v1.json')))
""")
md("## 10. Spatial block generation\nBlocks are 80.94 km grid cells. Repeat 1 uses grid origin (0, 0); repeats 2 and 3 use random origins (seed 20261118). Blocks are assigned to five folds with balanced household counts. The cell below regenerates the folds from this recipe and checks them against the fold file. The block size is the median of three outcome variogram ranges from an earlier analysis; the variogram table shows label-free alternatives.")
code("""
from common.folds import balanced_assign
rng = np.random.default_rng(20261118); ok = {}
for r in (1, 2, 3):
    ox, oy = (0.0, 0.0) if r == 1 else rng.uniform(0, BLOCK_KM, 2)
    b = (np.floor((df.x_km + ox) / BLOCK_KM).astype(int).astype(str) + '_' + np.floor((df.y_km + oy) / BLOCK_KM).astype(int).astype(str)).values
    sizes = pd.Series(b).value_counts().to_dict(); asg = balanced_assign(list(sizes), sizes, 5, rng)
    ok[r] = bool((b == df[f'block_r{r}'].astype(str).values).all() and (pd.Series(b).map(asg).values == df[f'spatial_fold_r{r}'].values).all())
print('folds regenerated exactly:', ok); display(pd.read_csv(REV / 'design_checks/D4_variogram_ranges_v1.csv').round(1))
""")
md("## 11. Spatial cross-validation\nFifteen development splits (3 repeats x 5 folds). Inner folds for calibration, thresholds and ensemble ranking are block-grouped inside each training set.")
code("""
display(pd.read_csv(REV / 'design_checks/D1_fold_structure_v1.csv')); print('inner folds:', s['D2'])
""")
md("## 12. ExtraTrees experiments (FS1 vs FS5)\nExtraTrees-300 (min_samples_leaf 10, max_features sqrt, seed 20261119) on FS1 and FS5, with outer test predictions and inner out-of-fold predictions.")
code("""
from paper1 import experiments as X, analysis as A
for r_, f_, *_ in dev_splits(df): X.e2_split(r_, f_)
print('E2 checkpoints:', len(list(X.E2.glob('*.npz'))))
runpy.run_path('src/paper1/rev_figures_v2.py', run_name='__main__'); show('Fig4_FS1_vs_FS5_paired_folds_v2')
""")
md("## 13. Machine-learning model comparisons\nThe 23-model library (fixed hyperparameters, corrected neighbourhood features). Spatial fold-mean ROC-AUC per model.")
code("""
for i in range(5, 20): X.e3_split(i)
from paper1 import rev_analysis_v1 as RA
RA.table_library()
""")
md("## 14. Ensemble construction\n(a) equal-weight rank average of the 23 models, with each model's test score converted to its empirical quantile in that model's inner out-of-fold scores; (b) the single model with the best inner out-of-fold ROC-AUC.")
code("""
# remaining experiment checkpoints (permutations, proximity/block size, temporal, importance), then rebuild all result tables
for job in [('household', -1)] + [(sch, k) for sch in ('household', 'village') for k in range(100)]: X.e5_part(*job)
for cond in X.E6_CONDITIONS:
    for r_ in (1, 2, 3):
        for f_ in range(5): X.e6_job(cond, r_, f_)
X.e7_temporal()
for r_, f_, *_ in dev_splits(df): X.e8_fold(r_, f_)
FM, BS = A.run_all(); P = A.collect_predictions()
print('ensemble (b): most often selected library model per disease (index in the 23-model list)')
print(P[P.model.str.startswith('(b)')].groupby('disease').selected_single.agg(lambda v: v.astype(int).value_counts().head(3).to_dict()))
""")
md("## 15. Model calibration\nPlatt scaling fitted on inner out-of-fold scores only (slope constrained to be non-negative; otherwise the training prevalence). Youden thresholds from the same inner data.")
code("""
FM.groupby(['disease', 'model'])[['Brier', 'calibration_intercept', 'calibration_slope']].median().round(3)
""")
md("## 16. ROC-AUC and PR-AUC\nDevelopment spatial performance (Table 4) and the comparison with random 5-fold CV run with the same corrected code.")
code("""
display(pd.read_csv(P1 / 'tables/Table4_overall_spatial_performance_v1.csv'))
from paper1 import rev_experiments_v1 as RX
for i in range(5): RX.e9_split(i)
from paper1 import rev_analysis_v1 as RA
T = RA.table_random_vs_spatial(); RA.fig_random_vs_spatial(T); display(RA.TABLES['Random vs spatial CV']); show('Fig3_random_vs_spatial_cv_v1'); show('FigS1_ROC_PR_curves_v1')
""")
md("## 17. Statistical significance testing\nPaired fold differences with the Nadeau-Bengio corrected resampled t-test; pooled differences with a location-cluster bootstrap (1,000 resamples); Holm correction across diseases. The second cell recomputes every statistic with independent code.")
code("""
display(pd.read_csv(P1 / 'tables/Table7_disease_specific_comparisons_v1.csv'))
""")
code("""
runpy.run_path('src/paper1/rev_verify_statistics_v1.py', run_name='__main__')
pd.read_csv(REV / 'statistics_verification/statistics_verification_summary_v1.csv')
""")
md("## 18. Permutation-based negative controls\nThe full pipeline, including the neighbourhood features, is rebuilt on permuted labels: 100 household-level and 100 village-level permutations (development repeat 1).")
code("""
display(pd.read_csv(P1 / 'tables/Table10_negative_controls_v1.csv')); runpy.run_path('src/paper1/fig5_permutation_v3.py'); show('Fig5_permutation_null_v3', P1 / 'figures')
""")
md("## 19. Proximity sensitivity\nTraining locations within 25, 50 or 100 km of any test location are removed; size-matched random removal is the control.")
code("""
display(pd.read_csv(P1 / 'tables/Table9_proximity_removal_v1.csv'))
""")
md("## 20. Block-size sensitivity\nThe analysis is repeated with 120 km and 160 km blocks; both the folds and the own-block exclusion use the larger blocks.")
code("""
display(pd.read_csv(P1 / 'tables/Table8_block_size_sensitivity_v1.csv')); show('Fig4_spatial_sensitivity_v2', P1 / 'figures')
""")
md("## 21. Within-year evaluation\nWithin-year ROC-AUC compares only households from the same survey year (Eq. 2). Campaign-specific models are trained on one campaign at a time and compared, on the same test households, with models trained on all campaigns.")
code("""
display(pd.read_csv(P1 / 'tables/Table5_within_year_performance_v1.csv')); display(pd.read_csv(P1 / 'tables/TableS_year_specific_AUC_v1.csv').pivot_table(index=['disease', 'model'], columns='year', values='AUC').round(3))
for r_ in (1, 2, 3):
    for f_ in range(5): RX.e10_split(r_, f_)
CT = RA.table_campaign_models(); display(RA.TABLES['Campaign vs pooled models']); show('Fig3_overall_vs_within_year_AUC_v1', P1 / 'figures'); runpy.run_path('src/paper1/rev_figures_v3.py', run_name='__main__'); show('Fig6_campaign_specific_vs_pooled_v3')
""")
md("## 22. Forward temporal validation\nTrain on 2012-2015 (1,564 households), test on 2018-2019 (1,776 households). This tests transfer to later campaigns, which is closer to forecasting than spatial CV.")
code("""
display(pd.read_csv(P1 / 'tables/Table12_forward_temporal_validation_v1.csv')); RA.fig_temporal(); show('Fig8_forward_temporal_v1')
""")
md("## 23. Feature importance\nGrouped permutation importance of FS5 ExtraTrees: drop in test ROC-AUC when one feature group is permuted (15 development folds).")
code("""
display(pd.read_csv(P1 / 'tables/TableS_grouped_permutation_importance_v1.csv')); runpy.run_path('src/paper1/figS4_importance_v2.py'); show('FigS4_grouped_importance_v2', P1 / 'figures')
""")
md("## 24. Final tables and figures\nAll tables are written as CSV and collected in one Excel workbook; all figures are saved as PNG (300 dpi) and PDF.")
code("""
runpy.run_path('src/paper1/fig1_workflow_v6.py'); RA.table_library(); RA.excel_workbook()
print(sorted(p.name for p in (REV / 'figures').glob('*.png'))); print(sorted(p.name for p in (REV / 'tables').glob('*')))
show('Fig1_workflow_v6')
""")
md("""
## Appendix. Completed analyses outside the scope of the revised paper (reported as observed)
1. **Confirmatory partition (2 x 5 folds).** Opened once on 2026-11-20 with the pre-correction code, for hypotheses registered just before opening. It was first reserved for a pre-registered optimisation method that failed its development gate.
2. **Population-based optimisation study (Stages 5-6).** Completed and pre-registered, but outside the scope of the revised paper. Its results are in the Supplementary Material (Section S6) and are not used in any result in the paper.
""")
code("""
print(json.dumps(json.load(open(ROOT / 'results/stage9_confirmatory_v1/hypothesis_results.json')), indent=1)[:1200])
display(pd.read_csv(P1 / 'tables/TableS_confirmatory_as_observed_confirmatory_summary_v1.csv'))
print('Stage 6 gate:', json.load(open(ROOT / 'results/stage6_v1/go_no_go_gate.json'))['diseases_passing'], 'of 3 diseases passed')
""")
nb['cells'] = C; nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}, 'language_info': {'name': 'python'}}
out = ROOT / 'IJIES_Paper1_Revised_Experiments_unexecuted_v1.ipynb'; nbf.write(nb, out); print('wrote', out, len(C), 'cells')
