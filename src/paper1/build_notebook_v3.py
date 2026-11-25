# v3: fixes the raw-file MD5 check (file_hashes takes one file) and adds Figure 5 v2. v2: Figure 1 from fig1_workflow_v5.py.
"""Builds IJIES_Paper1_Complete_Experiments.ipynb (run from the project root). The notebook imports src/paper1, so the code
that produced the paper's numbers and the notebook are the same code."""
import nbformat as nbf
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Spatial generalisation and survey-campaign confounding in machine-learning models for retrospective mapping of cattle disease occurrence in Ethiopia
**Complete experiments for the IJIES Paper 1 manuscript (v1, frozen configuration `configs/ijies_paper1_config_v1.yaml`).**

This notebook reproduces every table and figure in the manuscript from the raw data. Heavy steps write checkpoints under `results/paper1_v1/`. If a checkpoint is present it is loaded; delete `results/paper1_v1/` to recompute everything from scratch. Full recomputation takes about 3 hours on 2 CPU cores.

**Required data (not redistributed here).** The ILRI/MELDATA dataset (Gizaw, Dione, Wieland, Desta 2026, hdl:20.500.11766.1/FK2/8YFO3D, CC BY 4.0) must be placed in `data/raw/` as `cbpp.xlsx`, `fmd.xlsx` and `lsd.xlsx`, with the fold file `data/raw/stage3_folds_v1.csv`.
- Columns used: `ID`, `Location_id`, `Latitude`, `Longitude`, `Altitude`, `Temperature`, `Precipitation`, `Relative_humidity`, `Wind_speed`, `Year`, `Region`, and the outcome column of each file.
- Columns never used as predictors: `Weight_*`, `Year`, `Region`, `Year_region`, identifiers.

**Scope.** The models map household-reported disease occurrence retrospectively. They are **not** forecasting models, and the outcomes are not laboratory-confirmed infections. The locked confirmatory partition was opened once (2026-11-20) with the pre-correction code. Its results are reported as originally observed, and all new analyses use the development folds only.
""")
md("## 1. Environment and package versions")
code("""
import sys, os, json, platform, importlib
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / 'src')); os.environ['PYTHONPATH'] = str(ROOT / 'src'); os.chdir(ROOT)
import numpy as np, pandas as pd, matplotlib
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 70)
vers = {'python': platform.python_version()}
for m in ['numpy', 'pandas', 'scipy', 'sklearn', 'xgboost', 'lightgbm', 'catboost', 'matplotlib', 'joblib']:
    vers[m] = importlib.import_module(m).__version__
print(json.dumps(vers, indent=1))
import yaml; CFG = yaml.safe_load(open('configs/ijies_paper1_config_v1.yaml')); print('Frozen configuration loaded:', CFG['paper'][:80], '...')
""")
md("## 2. Data loading and validation")
code("""
from paper1.core import setup, dev_splits, DIS, FEATURE_SETS, OUT
from common.data import EXCLUDED, AUDITED_MD5, file_hashes
df, Y, NB, FB = setup()
print('households:', len(df), '| locations:', df['loc'].nunique(), '| survey years:', sorted(df.Year.unique()))
assert len(df) == 3340 and df['loc'].nunique() == 462
assert df.ID.is_unique, 'duplicate household IDs'
md5 = {f: file_hashes(ROOT / 'data/raw' / f)['md5'] for f in AUDITED_MD5}
print('raw-file MD5 match the audited values:', md5 == AUDITED_MD5); assert md5 == AUDITED_MD5
""")
code("""
# missing values, duplicates, coordinate consistency, block membership
pred_cols = ['Altitude', 'Temperature', 'Precipitation', 'Relative_humidity', 'Wind_speed', 'Latitude', 'Longitude']
checks = {'missing cells (predictors + outcomes)': int(df[pred_cols + DIS].isna().sum().sum()),
          'duplicate rows (predictors + outcomes)': int(df.duplicated(pred_cols + DIS).sum()),
          'locations with >1 coordinate pair': int((df.groupby('loc')[['Latitude', 'Longitude']].nunique() > 1).any(axis=1).sum()),
          'coordinate pairs shared by >1 location': int((df.groupby(['Latitude', 'Longitude'])['loc'].nunique() > 1).sum()),
          'locations in >1 81-km block (repeat 1)': int((df.groupby('loc').block_r1.nunique() > 1).sum()),
          'locations surveyed in >1 year': int((df.groupby('loc').Year.nunique() > 1).sum()),
          'latitude range': (float(df.Latitude.min()), float(df.Latitude.max())), 'longitude range': (float(df.Longitude.min()), float(df.Longitude.max()))}
for k, v in checks.items(): print(f'{k:45s} {v}')
assert checks['missing cells (predictors + outcomes)'] == 0 and checks['locations with >1 coordinate pair'] == 0
(OUT / 'tables').mkdir(parents=True, exist_ok=True); pd.Series({k: str(v) for k, v in checks.items()}).to_csv(OUT / 'tables' / 'data_validation_report_v1.csv')
""")
code("""
from paper1 import analysis as A
T1 = A.table1(); T1   # Table 1: dataset characteristics and reported prevalence by survey year
""")
md("## 3. Spatial feature engineering\nFS1 contains the five environmental variables. FS5 adds climate indices, coordinates, kernel and k-nearest-location disease rates for all three diseases (own and cross-disease), and sampling-density terms. Disease rates use **training households only**, and every query location excludes all training locations in its own 81 km block. k-nearest ties are broken by (distance rounded to 1e-6 km, location ID).")
code("""
A.table2()   # Table 2
""")
code("""
r, f, tr, te, bc = dev_splits(df)[0]
Xf, Xp, names = FB.build(FEATURE_SETS['FS5'], tr, te, df[bc].values, 1)
print('FS5 features for the FMD model:', len(names)); print(names)
fd = pd.DataFrame({'feature': names, 'uses_outcome_labels': [n.startswith(('kern', 'knn')) and not n.startswith('knndist') for n in names]})
fd.to_csv(OUT / 'tables' / 'feature_dictionary_v1.csv', index=False); fd.uses_outcome_labels.value_counts()
""")
code("""
# Provenance check 1: label-free features must not change when all labels are permuted
from paper1.core import NeighbourhoodV2
from common.features import climate_coords
from common.fitness import FeatureBuilder
rng = np.random.default_rng(20261121); Yp = Y[rng.permutation(len(Y))]
FBp = FeatureBuilder(df, Yp, NeighbourhoodV2(df, Yp), climate_coords(df))
Xf2, Xp2, _ = FBp.build(FEATURE_SETS['FS5'], tr, te, df[bc].values, 1)
free = [j for j, n in enumerate(names) if not (n.startswith(('kern', 'knn')) and not n.startswith('knndist'))]
assert np.array_equal(np.vstack([Xf, Xp])[:, free], np.vstack([Xf2, Xp2])[:, free]); print('label-free features unchanged under label permutation:', len(free), 'columns')
""")
code("""
# Provenance check 2: deterministic kNN tie-break verified against an independent loop-based reference implementation
# (full 15-split result from src/paper1/verify_tiebreak_v1.py; rerun that script to recompute, about 3 minutes)
print(json.dumps(json.load(open(OUT / 'verification' / 'P4_tiebreak_verification_v1.json')), indent=1))
""")
md("## 4. Spatial validation and leakage tests")
code("""
for r in (1, 2, 3):
    fo = df[f'spatial_fold_r{r}']; print(f'repeat {r}: blocks = {df[f"block_r{r}"].nunique()}, households per fold =', fo.value_counts().sort_index().tolist(),
          '| FMD positives per fold =', df.groupby(fo)['FMD'].sum().tolist())
""")
code("""
# Leakage test: flipping every test-fold label must leave all features and all predictions unchanged
from paper1.core import extratrees
Y2 = Y.copy(); Y2[te] = 1 - Y2[te]
FB2 = FeatureBuilder(df, Y2, NeighbourhoodV2(df, Y2), climate_coords(df))
a = FB.build(FEATURE_SETS['FS5'], tr, te, df[bc].values, 1); b = FB2.build(FEATURE_SETS['FS5'], tr, te, df[bc].values, 1)
assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
pa = extratrees().fit(a[0], Y[tr, 1]).predict_proba(a[1])[:, 1]; pb = extratrees().fit(b[0], Y2[tr, 1]).predict_proba(b[1])[:, 1]
assert np.array_equal(pa, pb); print('test-label flip: features and predictions bit-identical')
# flipping one training household never changes its own features or those of its 81 km block
h = int(tr[100]); Y3 = Y.copy(); Y3[h] = 1 - Y3[h]
c = FeatureBuilder(df, Y3, NeighbourhoodV2(df, Y3), climate_coords(df)).build(FEATURE_SETS['FS5'], tr, te, df[bc].values, 1)
same_block = np.where(df[bc].values[tr] == df[bc].values[h])[0]
assert np.array_equal(a[0][same_block], c[0][same_block]); print('training-household flip: own block unchanged,', int((a[0] != c[0]).any(1).sum()), 'rows outside the block changed')
""")
md("## 5. Machine-learning experiments (development folds)\nExtraTrees-300 with FS1 and FS5 (outer test and inner out-of-fold predictions), and the corrected 23-model library. These are checkpointed under `results/paper1_v1/E2_*` and `E3_*`.")
code("""
from joblib import Parallel, delayed
from paper1 import experiments as X
_ = Parallel(n_jobs=2)(delayed(X.e2_split)(r, f) for r in (1, 2, 3) for f in range(5))
_ = Parallel(n_jobs=2)(delayed(X.e3_split)(i) for i in range(5, 20))
print(len(list(X.E2.glob('*.npz'))), 'FS1/FS5 splits;', len(list(X.E3.glob('*.npz'))), 'library splits')
A.table3()   # Table 3: models and fixed hyperparameters
""")
md("## 6. Final ensemble, calibration and thresholds\n(a) is the equal-weight ECDF rank average of the 23 models, and (b) is the library model with the best inner-fold AUC. For every model:\n- Platt calibration (slope constrained to be at least 0) and the Youden threshold are fitted on inner out-of-fold predictions of the training set only.\n- Pooled discrimination uses each test score's quantile in the training out-of-fold distribution.")
code("""
P = A.collect_predictions(); FM = A.fold_metrics(P); (OUT / 'metrics').mkdir(exist_ok=True); FM.to_csv(OUT / 'metrics' / 'fold_metrics_v2.csv', index=False)
BS = A.bootstrap_table(P); BS.to_csv(OUT / 'metrics' / 'bootstrap_pooled_v2.csv', index=False)
T5 = A.table4_5(FM, BS); pd.read_csv(OUT / 'tables' / 'Table4_overall_spatial_performance_v1.csv')   # Table 4
""")
md("## 7. Survey-year analysis")
code("""
T5   # Table 5: overall vs within-survey-year ROC-AUC
""")
code("""
A.per_year_auc(P).pivot_table(index=['disease', 'model'], columns='year', values='AUC').round(3)   # year-specific AUC (supplement)
""")
code("""
A.table6_7(FM, BS); pd.read_csv(OUT / 'tables' / 'Table7_disease_specific_comparisons_v1.csv')   # Tables 6-7 (Holm across diseases)
""")
code("""
X.e7_temporal(); A.table12()   # Table 12: forward temporal validation (train 2012-2015, test 2018-2019)
""")
md("## 8. Sensitivity analyses and negative controls")
code("""
_ = Parallel(n_jobs=2)(delayed(X.e6_job)(c, r, f) for c in X.E6_CONDITIONS for r in (1, 2, 3) for f in range(5))
T9, A6 = A.table8_9(); pd.read_csv(OUT / 'tables' / 'Table8_block_size_sensitivity_v1.csv')   # Table 8
""")
code("""
T9   # Table 9: proximity removal and size-matched removal
""")
code("""
_ = Parallel(n_jobs=2)(delayed(X.e5_part)(s, p) for s, p in [('household', -1)] + [(s, p) for p in range(100) for s in ('household', 'village')])
T10, M = A.table10(); T10   # Table 10: negative controls (full pipeline rebuilt on permuted labels)
""")
code("""
A.table11()   # Table 11: ensemble sensitivity to the prior correction and the deterministic tie-break
""")
code("""
_ = Parallel(n_jobs=2)(delayed(X.e8_fold)(r, f) for r in (1, 2, 3) for f in range(5))
I = A.supplementary_tables(); I.pivot_table(index='group', columns='disease', values='mean_AUC_drop').round(4)   # grouped permutation importance
""")
md("## 9. Earlier results reported as observed\nThe confirmatory partition (opened once, pre-correction code) and the swarm-optimisation study (Stages 5-6) were run before the two code corrections. They are shown unchanged.")
code("""
print(json.dumps(json.load(open('results/stage9_confirmatory_v1/hypothesis_results.json')), indent=1)[:1500])
pd.read_csv('results/stage5_v1/optimiser_real_summary.csv')
""")
md("## 10. Figures")
code("""
import runpy; runpy.run_path('src/paper1/fig1_workflow_v5.py')   # Figure 1 (v5 layout; v1 labels overflowed)
A.fig_years(); A.fig_overall_vs_within(BS); A.fig_sensitivity(A6); A.fig_permutation(M); runpy.run_path('src/paper1/fig5_permutation_v2.py')   # Figure 5 v2 (integer counts, p in panels)
A.fig_roc_pr(P); A.fig_calibration(P); A.fig_confusion(P); A.fig_importance(I)
from IPython.display import Image, display
for n in ['Fig1_workflow_v5', 'Fig2_survey_years_v2', 'Fig3_overall_vs_within_year_AUC_v1', 'Fig4_spatial_sensitivity_v2', 'Fig5_permutation_null_v2',
          'FigS1_ROC_PR_curves_v1', 'FigS2_calibration_v1', 'FigS3_confusion_matrices_v1', 'FigS4_grouped_importance_v1']:
    print(n); display(Image(filename=str(OUT / 'figures' / f'{n}.png'), width=900))
""")
nb['cells'] = C
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}, 'language_info': {'name': 'python'}}
out = ROOT / 'IJIES_Paper1_Complete_Experiments_v3_unexecuted.ipynb'; nbf.write(nb, out); print('wrote', out)
