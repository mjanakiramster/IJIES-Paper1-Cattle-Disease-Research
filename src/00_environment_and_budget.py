"""Stage 0 — environment, file verification, inventory of earlier artifacts, learner timing and compute budget.
Run from the project root:  python src/00_environment_and_budget.py"""
import sys, os, json, time, platform, warnings, importlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.data import load, labels, file_hashes, AUDITED_MD5, DIS, SEED
from common.features import climate_coords, Neighbourhood
warnings.filterwarnings('ignore')

OUT = Path('results/stage0_v1'); OUT.mkdir(parents=True, exist_ok=False)   # versioned; never overwrite

# ---------- 1. environment ----------
libs = ['numpy', 'pandas', 'scipy', 'sklearn', 'xgboost', 'lightgbm', 'catboost', 'shap', 'matplotlib', 'openpyxl', 'joblib', 'imblearn', 'yaml', 'pytest']
ver = {}
for m in libs:
    try: ver[m] = importlib.import_module(m).__version__
    except Exception as e: ver[m] = f'MISSING ({type(e).__name__})'
env = {'python': sys.version.split()[0], 'platform': platform.platform(), 'cpu_cores': os.cpu_count(), 'libraries': ver}
with open(OUT / 'environment_report.txt', 'w') as f:
    f.write('ENVIRONMENT REPORT (Stage 0)\n'); f.write(json.dumps(env, indent=1))
print(json.dumps(env, indent=1))

# ---------- 2. dataset verification ----------
ver_rows = {}
for name in ['cbpp.xlsx', 'fmd.xlsx', 'lsd.xlsx', 'stage3_folds_v1.csv']:
    h = file_hashes(Path('data/raw') / name)
    ver_rows[name] = {**h, 'audited_md5': AUDITED_MD5.get(name), 'matches_audit': (AUDITED_MD5.get(name) == h['md5']) if name in AUDITED_MD5 else None}
json.dump(ver_rows, open(OUT / 'dataset_verification.json', 'w'), indent=1)
print({k: v['matches_audit'] for k, v in ver_rows.items()})

# ---------- 3. inventory (earlier artifacts on the user's computer, listed in Stage 0) ----------
inv = pd.read_csv('configs/earlier_artifacts_inventory.csv')
inv.to_csv(OUT / 'project_inventory.csv', index=False)

# ---------- 4. timing on a representative training set (development spatial fold r1/f0) ----------
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

df = load('data/raw'); Y = labels(df); NB = Neighbourhood(df, Y)
te = np.where(df.spatial_fold_r1.values == 0)[0]; tr = np.where(df.spatial_fold_r1.values != 0)[0]
C = climate_coords(df).values
t0 = time.perf_counter()
for _ in range(20): Ff, Fp, names = NB.rates(tr, te, df.block_r1.values)
t_feat = (time.perf_counter() - t0) / 20
Xf, Xp = np.column_stack([C[tr], Ff]), np.column_stack([C[te], Fp]); y, yt = Y[tr, 1], Y[te, 1]   # FMD as the example disease

learners = {
    'LR': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)),
    'GAM (splines + LR)': lambda: make_pipeline(StandardScaler(), SplineTransformer(n_knots=5, degree=3), LogisticRegression(C=0.5, max_iter=3000)),
    'RF 200 trees': lambda: RandomForestClassifier(200, min_samples_leaf=10, n_jobs=1, random_state=SEED),
    'ExtraTrees 200 trees': lambda: ExtraTreesClassifier(200, min_samples_leaf=10, n_jobs=1, random_state=SEED),
    'ExtraTrees 100 trees (search surrogate)': lambda: ExtraTreesClassifier(100, min_samples_leaf=10, n_jobs=1, random_state=SEED),
    'XGBoost 300': lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=1, verbosity=0, random_state=SEED),
    'LightGBM 300': lambda: LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=1, verbose=-1, random_state=SEED),
    'CatBoost 300': lambda: CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05, verbose=0, thread_count=1, random_seed=SEED, allow_writing_files=False),
    'SVM RBF (no Platt)': lambda: make_pipeline(StandardScaler(), SVC(C=1.0, gamma='scale')),
    'MLP (32,16)': lambda: make_pipeline(StandardScaler(), MLPClassifier((32, 16), alpha=1e-2, max_iter=400, random_state=SEED)),
}
rows = []
for name, mk in learners.items():
    ts, auc = [], np.nan
    for rep in range(3):
        t1 = time.perf_counter(); m = mk().fit(Xf, y)
        s = m.decision_function(Xp) if 'SVM' in name else m.predict_proba(Xp)[:, 1]
        ts.append(time.perf_counter() - t1); auc = roc_auc_score(yt, s)
    rows.append({'learner': name, 'level': 'household rows', 'n_train': len(tr), 'n_features': Xf.shape[1], 'sec_per_fit_median': float(np.median(ts))})
# location-level surrogate (one row per training location, rate as target)
lf = NB.loc[tr]; Xl = pd.DataFrame(Xf).groupby(lf).mean(); nl = pd.Series(lf).value_counts().reindex(Xl.index).values; yl = pd.Series(y).groupby(lf).mean().reindex(Xl.index).values
ts = []
for rep in range(3):
    t1 = time.perf_counter(); ExtraTreesRegressor(100, min_samples_leaf=3, n_jobs=1, random_state=SEED).fit(Xl.values, yl, sample_weight=nl).predict(Xp); ts.append(time.perf_counter() - t1)
rows.append({'learner': 'ExtraTrees 100 location-level (search surrogate, alt.)', 'level': 'locations', 'n_train': len(Xl), 'n_features': Xf.shape[1], 'sec_per_fit_median': float(np.median(ts))})
rows.append({'learner': 'Neighbourhood feature construction (one fit/predict set)', 'level': 'locations', 'n_train': NB.NL, 'n_features': len(names), 'sec_per_fit_median': t_feat})
T = pd.DataFrame(rows); T.to_csv(OUT / 'learner_timing.csv', index=False); print(T.round(4).to_string(index=False))

# ---------- 5. compute budget ----------
tt = dict(zip(T.learner, T.sec_per_fit_median))
t_et200, t_sur, t_loc, t_f = tt['ExtraTrees 200 trees'], tt['ExtraTrees 100 trees (search surrogate)'], tt['ExtraTrees 100 location-level (search surrogate, alt.)'], t_feat
cores = os.cpu_count() or 1
scen = [  # name, pop, iters, inner_fits, sec_per_inner_fit, outer, diseases, seeds, optimisers
    ('v7 as written: pop 20 x 30 it, 3x5 inner, ExtraTrees 200, 15 outer, 3 seeds, 7 optimisers', 20, 30, 15, t_et200 + t_f, 15, 3, 3, 7),
    ('Stage 5 optimiser comparison (reduced): pop 12 x 20 it, 2x5 inner, row-level surrogate, 5 outer (repeat 1), 3 seeds, 7 optimisers', 12, 20, 10, t_sur + t_f, 5, 3, 3, 7),
    ('Stage 5 optimiser comparison (reduced), location-level surrogate', 12, 20, 10, t_loc + t_f, 5, 3, 3, 7),
    ('Stage 6 SR-SGO development: pop 12 x 20 it, 2x5 inner, row-level surrogate, 15 outer, 1 seed', 12, 20, 10, t_sur + t_f, 15, 3, 1, 1),
    ('Stage 6 SR-SGO development, location-level surrogate', 12, 20, 10, t_loc + t_f, 15, 3, 1, 1),
    ('Stage 8 ablations: 8 variants of Stage 6 on 5 outer folds (repeat 1), row-level surrogate', 12, 20, 10, t_sur + t_f, 5, 3, 1, 8),
    ('Stage 9 confirmatory SR-SGO search: 10 confirmatory folds, row-level surrogate', 12, 20, 10, t_sur + t_f, 10, 3, 1, 1),
]
B = pd.DataFrame([{'scenario': s, 'population': p, 'iterations': it, 'fitness_evaluations_per_run': p * it, 'inner_fits_per_evaluation': k,
                   'sec_per_inner_fit_incl_features': round(sec, 4), 'outer_folds': o, 'diseases': d, 'seeds': sd, 'optimisers_or_variants': op,
                   'core_hours': p * it * k * sec * o * d * sd * op / 3600, 'wall_hours_on_available_cores': p * it * k * sec * o * d * sd * op / 3600 / cores}
                  for s, p, it, k, sec, o, d, sd, op in scen])
B.to_csv(OUT / 'compute_budget.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 110)
print(B[['scenario', 'fitness_evaluations_per_run', 'sec_per_inner_fit_incl_features', 'core_hours', 'wall_hours_on_available_cores']].round(2).to_string(index=False))
