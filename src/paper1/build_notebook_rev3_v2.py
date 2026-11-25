"""Builds the revision-2 notebook in the 22 sections requested for the study (Phase 5, item H), v2: Tables 1-2 and Figure 2 are
computed where they are first shown (v1 read them from files that a fresh run had not yet written).
Writes IJIES_Paper1_Experiments_22sections_unexecuted_v2.ipynb in the project root. Run from the project root:
python src/paper1/build_notebook_rev3_v2.py"""
import nbformat as nbf
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Spatial generalisation and survey-campaign confounding in machine-learning models for retrospective mapping of cattle disease occurrence in Ethiopia
**Executable notebook (22 sections).** Household-reported CBPP, FMD and LSD occurrence (ILRI MELDATA; 3,340 households, 462 locations, five survey campaigns, 2012-2019).

**What it does.** Every table and figure is computed from the raw data by the functions in `src/paper1/`. Each experiment writes a checkpoint under `results/`; if the checkpoint exists it is loaded, otherwise it is computed. Delete `results/paper1_v1`, `results/paper1_rev_v1` and `results/paper1_rev2_v1` to recompute everything. Inputs that are kept: the raw data, the fold file, the earlier 23-model library cache for the models that use no outcome-derived features (`results/stage4_v1`, `data/reference`), and the corrected-prior refits (`results/audit_v1/A5`).

**Scope.** Retrospective mapping of household-reported occurrence; not forecasting; not laboratory-confirmed. No optimisation or swarm algorithm is used in any analysis. `Weight_*`, `Year`, `Region`, `Year_region` and identifiers are never predictors (one library model uses year-region group sizes as training weights; see Section 10).
""")
md("## 1. Library installation and imports\nIn Google Colab, upload `IJIES_Paper1_code_and_results.zip` and the three data files, then run this cell. Locally, install `requirements.txt` first.")
code("""
import sys, os, subprocess, zipfile, json, platform, importlib, runpy, warnings
from pathlib import Path
if 'google.colab' in sys.modules:
    if Path('IJIES_Paper1_code_and_results.zip').exists() and not Path('IJIES_Paper1_code_and_results').exists():
        zipfile.ZipFile('IJIES_Paper1_code_and_results.zip').extractall('.')
    os.chdir('IJIES_Paper1_code_and_results')
    for f in ('cbpp.xlsx', 'fmd.xlsx', 'lsd.xlsx'):
        if Path('..', f).exists(): os.replace(Path('..', f), Path('data/raw', f))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / 'src')); os.environ['PYTHONPATH'] = str(ROOT / 'src'); os.chdir(ROOT)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display
warnings.filterwarnings('ignore'); pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40); pd.set_option('display.max_colwidth', 60)
vers = {'python': platform.python_version()}
for m in ['numpy', 'pandas', 'scipy', 'sklearn', 'xgboost', 'lightgbm', 'catboost', 'shap', 'matplotlib']:
    try: vers[m] = importlib.import_module(m).__version__
    except ImportError: vers[m] = 'NOT INSTALLED'
print(json.dumps(vers, indent=1)); assert vers['shap'] != 'NOT INSTALLED', 'install shap (requirements.txt) for Section 15'
P1, REV, R2 = ROOT / 'results/paper1_v1', ROOT / 'results/paper1_rev_v1', ROOT / 'results/paper1_rev2_v1'
def check(cond, msg):
    if not cond: raise AssertionError(msg)
    print('OK -', msg)
def show(name, width=900):
    for f in (R2 / 'figures', REV / 'figures', P1 / 'figures'):
        p = f / f'{name}.png'
        if p.exists(): display(Image(filename=str(p), width=width)); return
    print('missing figure:', name)
def tab(path): return pd.read_csv(path)
""")
md("## 2. Dataset loading\nThe three disease files share household IDs, environmental variables, coordinates and the year-region code. `setup()` checks that these columns are identical across files, merges them, attaches the fixed fold file and projects coordinates to km.")
code("""
from paper1.core import setup, dev_splits, DIS, FEATURE_SETS, auc
from common.data import AUDITED_MD5, file_hashes, BLOCK_KM
for f in AUDITED_MD5: check((ROOT / 'data/raw' / f).exists(), f'raw file present: {f}')
md5 = {f: file_hashes(ROOT / 'data/raw' / f)['md5'] for f in AUDITED_MD5}; check(md5 == AUDITED_MD5, 'raw-file MD5 hashes match the audited values')
df, Y, NB, FB = setup()
check(len(df) == 3340 and df['loc'].nunique() == 462, '3,340 households at 462 locations')
df[['ID', 'Location_id', 'Year', 'Region', 'Latitude', 'Longitude', 'Altitude', 'Temperature', 'Precipitation', 'Relative_humidity', 'Wind_speed'] + DIS].head()
""")
md("## 3. Data quality assessment\nMissing values, duplicates, coordinate consistency, outcome coding, and a flag for 2012 households labelled Afar but located south of 8 degrees N (kept as published; region is not a predictor).")
code("""
pred = ['Altitude', 'Temperature', 'Precipitation', 'Relative_humidity', 'Wind_speed', 'Latitude', 'Longitude']
q = {'missing cells (predictors, outcomes, year, region)': int(df[pred + DIS + ['Year', 'Region']].isna().sum().sum()), 'duplicate household IDs': int(df.ID.duplicated().sum()),
     'duplicate rows (predictors + outcomes)': int(df.duplicated(pred + DIS).sum()), 'locations with >1 coordinate pair': int((df.groupby('loc')[['Latitude', 'Longitude']].nunique() > 1).any(axis=1).sum()),
     'outcome values other than 0/1': int((~df[DIS].isin([0, 1])).sum().sum()), 'locations surveyed in >1 year': int((df.groupby('loc').Year.nunique() > 1).sum()),
     '2012 Afar households south of 8 deg N (flag)': int(((df.Year == 2012) & (df.Region == 'Afar') & (df.Latitude < 8)).sum())}
display(pd.Series(q, name='count').to_frame())
check(q['missing cells (predictors, outcomes, year, region)'] == 0 and q['duplicate household IDs'] == 0 and q['outcome values other than 0/1'] == 0, 'no missing values, duplicate IDs or invalid outcomes')
pred_cols = pred; checks = {'missing cells (predictors + outcomes)': int(df[pred_cols + DIS].isna().sum().sum()), 'duplicate rows (predictors + outcomes)': int(df.duplicated(pred_cols + DIS).sum()),
          'locations with >1 coordinate pair': int((df.groupby('loc')[['Latitude', 'Longitude']].nunique() > 1).any(axis=1).sum()),
          'coordinate pairs shared by >1 location': int((df.groupby(['Latitude', 'Longitude'])['loc'].nunique() > 1).sum()),
          'locations in >1 81-km block (repeat 1)': int((df.groupby('loc').block_r1.nunique() > 1).sum()),
          'locations surveyed in >1 year': int((df.groupby('loc').Year.nunique() > 1).sum()),
          'latitude range': (float(df.Latitude.min()), float(df.Latitude.max())), 'longitude range': (float(df.Longitude.min()), float(df.Longitude.max()))}
(P1 / 'tables').mkdir(parents=True, exist_ok=True); pd.Series({k: str(v) for k, v in checks.items()}).to_csv(P1 / 'tables' / 'data_validation_report_v1.csv')
""")
md("## 4. Exploratory data analysis\nPrevalence by survey year, overlap between diseases, environmental differences between positive and negative households, and survey campaigns (coverage, overlap and within-campaign spread).")
code("""
from paper1 import analysis as A
display(A.table1())
print('household-level phi correlation between diseases'); display(pd.DataFrame(np.corrcoef(Y.T), index=DIS, columns=DIS).round(3))
smd = pd.DataFrame({dn: [(df.loc[Y[:, d] == 1, v].mean() - df.loc[Y[:, d] == 0, v].mean()) / df[v].std() for v in pred[:5]] for d, dn in enumerate(DIS)}, index=pred[:5])
display(df[pred[:5]].describe().T.round(2)); display(smd.round(3))
from paper1 import experiments as X; X.e2_split(1, 0)   # the test-label isolation check (D3) compares against this checkpoint
runpy.run_path('src/paper1/rev_design_checks_v1.py', run_name='__main__')
display(tab(REV / 'design_checks/D5_campaign_descriptors_v1.csv')); display(tab(REV / 'design_checks/D5_variance_shares_v1.csv').round(3)); A.fig_years(); show('Fig2_survey_years_v2')
""")
md("## 5. Preprocessing\nNo imputation is needed. Tree models use raw values; linear, distance and kernel models standardise inputs inside each training set. `Weight_*` closely follows the share of same-outcome households at a location, so it is excluded; the check below shows why.")
code("""
from common.data import EXCLUDED
print('excluded from predictors:', EXCLUDED)
display(tab(ROOT / 'results/stage1_v1/weight_leakage_check.csv'))
""")
md("## 6. Feature engineering\nFS1 = 5 environmental variables; FS2 = FS1 + 6 climate indices (11); FS3 = FS2 + 4 coordinate terms (15); FS4 = FS3 + 9 kernel and 9 k-nearest-location disease rates for all three diseases (33); FS5 = FS4 + 6 sampling-density terms (39). Rates use training rows only and exclude the query location's own 81 km block.")
code("""
from paper1.rev2_experiments_v1 import ABLATION
r, f, tr, te, bc = dev_splits(df)[0]; counts = {}
for name, spec in {'FS1': FEATURE_SETS['FS1'], **ABLATION, 'FS5': FEATURE_SETS['FS5']}.items():
    _, _, nm = FB.build(spec, tr, te, df[bc].values, 1); counts[name] = len(nm)
print(counts); check(counts == {'FS1': 5, 'FS2': 11, 'FS3': 15, 'FS4': 33, 'FS5': 39}, 'feature counts FS1-FS5 = 5, 11, 15, 33, 39')
Xf, Xp, names = FB.build(FEATURE_SETS['FS5'], tr, te, df[bc].values, 1)
fd = pd.DataFrame({'feature': names, 'uses_outcome_labels': [n.startswith(('kern', 'knn')) and not n.startswith('knndist') for n in names]})
fd.to_csv(P1 / 'tables' / 'feature_dictionary_v1.csv', index=False); display(fd.T)
display(A.table2())
""")
md("## 7. Spatial fold generation\nBlocks are 80.94 km grid cells (repeat 1 origin (0, 0), repeats 2-3 random origins, seed 20261118), assigned to five folds with balanced household counts. The block size is the median of three outcome-variogram ranges from an earlier analysis (a label-derived choice; see the variogram table). The cell regenerates the folds and checks them against the fold file, then shows train-test distances under each scheme.")
code("""
from common.folds import balanced_assign
rng = np.random.default_rng(20261118); ok = {}
for r in (1, 2, 3):
    ox, oy = (0.0, 0.0) if r == 1 else rng.uniform(0, BLOCK_KM, 2)
    b = (np.floor((df.x_km + ox) / BLOCK_KM).astype(int).astype(str) + '_' + np.floor((df.y_km + oy) / BLOCK_KM).astype(int).astype(str)).values
    sizes = pd.Series(b).value_counts().to_dict(); asg = balanced_assign(list(sizes), sizes, 5, rng)
    ok[r] = bool((b == df[f'block_r{r}'].astype(str).values).all() and (pd.Series(b).map(asg).values == df[f'spatial_fold_r{r}'].values).all())
check(all(ok.values()), f'folds regenerated exactly from the recipe {ok}')
display(tab(REV / 'design_checks/D1_fold_structure_v1.csv')); display(tab(REV / 'design_checks/D4_variogram_ranges_v1.csv').round(1))
""")
md("## 8. Leakage-safe feature construction\nAll outcome-derived features are rebuilt inside each training split. Checks: test-label flips leave features, predictions, calibration and thresholds unchanged (FS1-FS5); training-label flips never change the own block; SHAP, AOA and campaign-identification outputs do not depend on test labels; the k-nearest tie rule matches an independent implementation.")
code("""
s = json.load(open(REV / 'design_checks/design_checks_summary_v1.json')); display(pd.DataFrame(s['D3']))
runpy.run_path('src/paper1/rev2_leakage_checks_v1.py', run_name='__main__')
L = json.load(open(R2 / 'leakage_checks/rev2_leakage_checks_v1.json'))
check(max(x['max_feature_change'] + x['max_prediction_change'] for x in L['L1_test_label_flip']) == 0, 'FS2-FS4 unchanged by test-label flips')
check(L['L2_training_flip_own_block_max_change'] == 0, 'training-label flips never change own-block features')
check(max(L['L4_downstream_analyses_test_label_flip'].values()) == 0, 'SHAP, AOA and year identification unchanged by test-label flips')
if not (P1 / 'verification/P4_tiebreak_verification_v1.json').exists(): runpy.run_path('src/paper1/verify_tiebreak_v1.py', run_name='__main__')
print(json.load(open(P1 / 'verification/P4_tiebreak_verification_v1.json')))
""")
md("## 9. Machine-learning model training\nAll experiment checkpoints: E2 FS1/FS5, E3 23-model library, E5 permutations, E6 proximity/block size, E7 temporal, E8 grouped importance, E9 random CV, E10/E10b campaign-specific models, E11 ablation, E12 SHAP and partial dependence, E13 area of applicability, E14 campaign identification. Missing checkpoints are computed (about 4 hours on 2 cores in total).")
code("""
from paper1 import experiments as X, analysis as A, rev_experiments_v1 as RX, rev2_experiments_v1 as R2X
for r_, f_, *_ in dev_splits(df): X.e2_split(r_, f_); X.e8_fold(r_, f_)
for i in range(5, 20): X.e3_split(i)
for job in [('household', -1)] + [(sch, k) for sch in ('household', 'village') for k in range(100)]: X.e5_part(*job)
for cond in X.E6_CONDITIONS:
    for r_ in (1, 2, 3):
        for f_ in range(5): X.e6_job(cond, r_, f_)
X.e7_temporal()
for i in range(5): RX.e9_split(i)
for r_ in (1, 2, 3):
    for f_ in range(5):
        RX.e10_split(r_, f_); R2X.e10b_split(r_, f_); R2X.e11_split(r_, f_); R2X.e13_split(r_, f_); R2X.e14_split(r_, f_)
for f_ in range(5): R2X.e12_split(1, f_)
counts = {p.name: len(list(p.glob('*'))) for p in [X.E2, X.E3, X.E5, X.E6, X.E8, RX.E9, RX.E10, R2X.E10b, R2X.E11, R2X.E12, R2X.E13, R2X.E14]}
print(counts); check(counts['E5_permutations'] >= 201 and counts['E11_ablation'] >= 15, 'all experiment checkpoints present')
FM, BS = A.run_all(); P = A.collect_predictions()
""")
md("## 10. Hyperparameter documentation\nRead directly from the instantiated model objects (not typed by hand). All hyperparameters were fixed in advance; no tuning on test folds. Note the group-balanced CatBoost model, which weights training households by the inverse size of their year-region group.")
code("""
from paper1 import rev2_reporting_v1 as RR
H = RR.hyperparameter_table(); H
""")
md("## 11. Random cross-validation\nRandom 5-fold CV at household level, with the same corrected code. Households from the same location fall in both training and test sets, so random CV measures interpolation within surveyed villages, not prediction for new areas.")
code("""
from paper1 import rev_analysis_v1 as RA
T = RA.table_random_vs_spatial(); RA.fig_random_vs_spatial(T); display(RA.TABLES['Random vs spatial CV']); show('Fig3_random_vs_spatial_cv_v1')
""")
md("## 12. Spatial block cross-validation\nMain results (81 km blocks, 15 development splits), and the sensitivity of the FS5 gain to proximity buffers and to 120/160 km blocks. The sensitivity analyses are exploratory.")
code("""
display(tab(P1 / 'tables/Table4_overall_spatial_performance_v1.csv')); display(tab(P1 / 'tables/Table9_proximity_removal_v1.csv')); display(tab(P1 / 'tables/Table8_block_size_sensitivity_v1.csv'))
show('Fig4_spatial_sensitivity_v2')
""")
md("## 13. Feature ablation experiments\nFS1 -> FS5 on identical folds, inner folds and seeds. Each step adds one feature group; differences are tested with the corrected resampled t-test (Holm over 12 tests) and a location-cluster bootstrap of pooled predictions.")
code("""
from paper1 import rev2_analysis_v1 as RV
TA, IA = RV.ablation(); display(TA.drop(columns=[c for c in TA if c.startswith('_')])); display(IA.drop(columns=[c for c in IA if c.startswith('_')])); show('R2Fig1_ablation_v1')
""")
md("## 14. Model comparison\nAll 23 library models, the rank-average ensemble (a), the inner-selected single model (b), and the ensemble without the group-balanced CatBoost model.")
code("""
display(RA.table_library()); display(tab(P1 / 'tables/Table7_disease_specific_comparisons_v1.csv')); display(RV.ensemble_without_gb())
""")
md("## 15. Explainability\nGrouped permutation importance (15 folds) and SHAP values with partial dependence (FS5 ExtraTrees, repeat 1). Models are fitted on training rows only; SHAP is computed for test rows, partial dependence on training rows. These are predictive associations, not causal effects.")
code("""
display(tab(P1 / 'tables/TableS_grouped_permutation_importance_v1.csv'))
S, G, rho = RV.shap_pdp(); display(G.round(4)); print('Spearman rho, SHAP vs permutation group ranking:', {k: round(v, 2) for k, v in rho.items()})
show('R2Fig2_shap_top_features_v1'); show('R2Fig3_partial_dependence_v1')
""")
md("## 16. Spatial applicability\nArea of applicability (Meyer and Pebesma 2021) for test locations under spatial CV: importance-weighted standardised distance to the nearest training household, divided by the mean training distance; the threshold comes from inner spatial folds. Environmental values exist only at surveyed locations, so the maps show surveyed locations, not a continuous surface. The dissimilarity index is not a risk or certainty measure.")
code("""
display(RV.aoa()); show('R2Fig4_area_of_applicability_FS5_v1')
""")
md("## 17. Calibration\nDiscrimination and calibration are reported separately. Pooled predictions per repeat; 95% intervals from 1,000 location-cluster bootstrap resamples.")
code("""
display(RV.metric_cis()); show('R2Fig5_calibration_with_bands_v1')
""")
md("## 18. Survey-campaign analysis\nWithin-year AUC, year-specific AUC, campaign-specific versus pooled models, and whether the features (especially the neighbourhood rates) identify the survey campaign.")
code("""
display(tab(P1 / 'tables/Table5_within_year_performance_v1.csv')); show('Fig3_overall_vs_within_year_AUC_v1')
CT = RA.table_campaign_models(); display(RA.TABLES['Campaign vs pooled models']); runpy.run_path('src/paper1/rev_figures_v3.py', run_name='__main__'); show('Fig6_campaign_specific_vs_pooled_v3')
print('max |difference| between saved campaign predictions (E10b) and the original E10 AUCs:', RV.check_e10b())
CI, diff = RV.campaign_identification(); display(CI); print('neighbourhood rates minus FS1, macro AUC for survey year [corrected-t 95% CI]: %+.3f [%+.3f, %+.3f], p = %.4f' % diff)
""")
md("## 19. Temporal validation\nTrain 2012-2015 (1,564 households), test 2018-2019 (1,776). This is the closest test to forecasting; poor results are reported as they are.")
code("""
display(tab(P1 / 'tables/Table12_forward_temporal_validation_v1.csv')); RA.fig_temporal(); show('Fig8_forward_temporal_v1')
""")
md("## 20. Statistical testing\nCorrected resampled t-test (overlapping training sets across folds), location-cluster bootstrap (households in the same location are not independent), Holm correction, and permutation negative controls that rebuild the whole pipeline. The second cell recomputes every reported statistic with independent code.")
code("""
display(tab(P1 / 'tables/Table7_disease_specific_comparisons_v1.csv')); display(tab(P1 / 'tables/Table10_negative_controls_v1.csv'))
runpy.run_path('src/paper1/fig5_permutation_v3.py'); show('Fig5_permutation_null_v3')
""")
code("""
runpy.run_path('src/paper1/export_predictions_v1.py', run_name='__main__'); runpy.run_path('src/paper1/rev_verify_statistics_v1.py', run_name='__main__')
V = tab(REV / 'statistics_verification/statistics_verification_summary_v1.csv'); display(V)
check(V[V.quantity.str.contains('Table')].max_abs_difference.max() < 0.001, 'all table statistics reproduced by independent code (within rounding)')
""")
md("## 21. Figures and tables\nAll remaining figures (PNG 300 dpi and PDF).")
code("""
runpy.run_path('src/paper1/fig1_workflow_v6.py'); runpy.run_path('src/paper1/rev_figures_v2.py', run_name='__main__'); runpy.run_path('src/paper1/figS2_calibration_v2.py'); runpy.run_path('src/paper1/figS4_importance_v2.py')
RV.distance_distributions(); show('R2Fig6_train_test_distances_v1'); show('Fig1_workflow_v6'); show('Fig4_FS1_vs_FS5_paired_folds_v2')
""")
md("## 22. Exporting final results\nFold assignments (all schemes) and household-level predictions of every experiment are written to CSV; all result tables are collected in an Excel workbook.")
code("""
print(RR.export_all()); RA.excel_workbook()
with pd.ExcelWriter(R2 / 'tables/IJIES_Paper1_revision2_tables_v1.xlsx') as W:
    for p in sorted((R2 / 'tables').glob('*.csv')): tab(p).to_excel(W, sheet_name=p.stem[:31], index=False)
for d in (P1 / 'tables', REV / 'tables', R2 / 'tables', R2 / 'figures', R2 / 'exports'): print(d.relative_to(ROOT), len(list(d.glob('*'))), 'files')
""")
md("""
## Appendix. Completed analyses reported as observed (not used in the revised analysis)
1. **Confirmatory partition (2 x 5 folds):** opened once on 2026-11-20 with the pre-correction code, for hypotheses registered just before opening.
2. **Pre-registered optimisation study:** completed before this revision and outside its scope. It is reported in Supplementary Section S2. No optimisation algorithm is used in this notebook.
""")
code("""
print(json.dumps(json.load(open(ROOT / 'results/stage9_confirmatory_v1/hypothesis_results.json')), indent=1)[:900])
""")
nb['cells'] = C; nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}, 'language_info': {'name': 'python'}}
out = ROOT / 'IJIES_Paper1_Experiments_22sections_unexecuted_v2.ipynb'; nbf.write(nb, out); print('wrote', out, len(C), 'cells')
