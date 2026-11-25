# %% [markdown]
# # Cattle disease prediction — run 3: a robust swarm ensemble that aims to beat every single model
#
# **Why run 3.** In run 2 the swarm-optimised ensemble (SA-HGWOPSO) did not beat simpler methods: a plain average of all models ranked first, and on each disease some single model scored higher than the ensemble. Two reasons were visible in the run-2 results: (i) the optimised weights over-fitted the inner folds (all optimisers reached the same inner fitness but did no better on the test folds), and (ii) the library had little diversity (mostly tree models on the same features).
#
# **What run 3 changes (same dataset, exactly the same test folds as runs 1 and 2):**
# 1. **More diverse learners.** A geostatistical learner (location-level Gaussian-process *kriging* on empirical logits, Matérn kernel), a smooth GAM without neighbourhood rates, and models with **adaptive k-nearest-village outbreak rates** (fixed-radius rates are empty where sampling is sparse).
# 2. **Proposed method — RB-HGWOPSO (robust, bagged hybrid GWO–PSO ensemble selection):**
#    * every model's scores are put on a common rank scale (empirical CDF of its own training out-of-fold scores);
#    * the fitness rewards weights that work in **every** inner spatial block: mean block ROC-AUC − 0.5 × SD across blocks (ties scored correctly);
#    * the optimiser is run on 10 bootstrap resamples of locations and the weight vectors are averaged (bagging), which reduces over-fitting.
# 3. **Pre-registered success rule** (below) and every variant reported, so nothing is cherry-picked.
#
# The proposed method is compared with all 21 single models, simple average (probabilities and ranks), Caruana selection, stacking, the run-2 SA-HGWOPSO, and the same robust bagged scheme driven by PSO, GWO and DE (equal budget). Ablations remove bagging, the robust fitness, the rank scaling, and the new learners.
#
# **Primary score:** ROC-AUC under spatial-block CV (3 repeats × 5 folds). Leave-one-region-out and random CV are also reported. Everything is saved in a new `results_vN` folder.

# %%
# %pip install xgboost catboost scikit-learn scipy pandas matplotlib openpyxl joblib
import os, sys, json, time, itertools, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats, optimize
from joblib import Parallel, delayed
import sklearn, xgboost, catboost
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
warnings.filterwarnings('ignore')
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40); pd.set_option('display.precision', 3)
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

FAST = os.environ.get('CATTLE_FAST') == '1'
SEED = 20261119; N_JOBS = 2; K = 5
N_REP = 1 if FAST else 3
SEEDS = [0] if FAST else [0, 1, 2]
POP, ITERS = (10, 5) if FAST else (20, 30)       # 600 fitness evaluations per optimiser run
N_BAG = 3 if FAST else 10                         # bootstrap bags for the robust bagged ensembles
LAM_SD = 0.5                                      # penalty on the spread of block ROC-AUC (fixed in advance)
B_BOOT = 50 if FAST else 400
DATA = Path('data'); COL = {'CBPP': '#2a78d6', 'FMD': '#eb6834', 'LSD': '#1baf7a'}

def next_free_dir(base):
    v = 1
    while Path(f'{base}_v{v}').exists(): v += 1
    p = Path(f'{base}_v{v}'); (p / 'figures').mkdir(parents=True); return p
OUT = next_free_dir('results'); FIG = OUT / 'figures'; VTAG = OUT.name.split('_')[-1]
def save_table(df, name, index=False):
    df.to_csv(OUT / f'{name}_{VTAG}.csv{".gz" if len(df) > 100000 else ""}', index=index); return df
def save_fig(fig, name):
    fig.savefig(FIG / f'{name}_{VTAG}.png', dpi=200, bbox_inches='tight'); plt.show()
VERSIONS = {'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
            'scikit-learn': sklearn.__version__, 'xgboost': xgboost.__version__, 'catboost': catboost.__version__}
print('Saving to:', OUT.resolve()); print(VERSIONS); print('FAST' if FAST else 'Full run')

# %% [markdown]
# ## 1. Data and the identical folds from run 1

# %%
DIS = ['CBPP', 'FMD', 'LSD']
ENV = ['Altitude', 'Temperature', 'Precipitation', 'Relative_humidity', 'Wind_speed']
raw = {d: pd.read_excel(DATA / f'{d.lower()}.xlsx', sheet_name=d.lower()).rename(columns={'wind': 'Wind_speed'}).sort_values('id').reset_index(drop=True) for d in DIS}
SH = ['id'] + ENV + ['Year_region', 'x', 'y']
for d in DIS[1:]: assert raw[d][SH].equals(raw['CBPP'][SH])
df = raw['CBPP'][SH].copy()
for d in DIS: df[d] = raw[d][d.lower()].astype(int).values
df = df.rename(columns={'id': 'ID', 'x': 'Longitude', 'y': 'Latitude'})
df[['Year', 'Region']] = df.Year_region.str.split('.', n=1, expand=True); df['Year'] = df.Year.astype(int)
folds = pd.read_csv(DATA / 'stage3_folds_v1.csv')
df = df.merge(folds.drop(columns=['Region', 'Year']), on='ID', how='left', validate='one_to_one')
assert df.Location_id.notna().all() and len(df) == 3340
df['loc'] = df.Location_id.astype(int) - 1
Y_ALL = df[DIS].values.astype(int); N = len(df)
print(f'{N} households, {df["loc"].nunique()} locations; folds loaded from run 1')

# %% [markdown]
# ## 2. Pre-specified rules (written before running)

# %%
elig = {}
for r in df.Region.unique():
    for d in DIS:
        pos = int(df.loc[df.Region == r, d].sum()); elig[(r, d)] = pos >= 10 and (int((df.Region == r).sum()) - pos) >= 10
LORO_REGIONS = [r for r in sorted(df.Region.unique()) if any(elig[(r, d)] for d in DIS)]
PROPOSED = 'RB-HGWOPSO ensemble (proposed)'
RULES = {'primary_metric': 'ROC-AUC', 'secondary': ['PR-AUC', 'Brier'], 'primary_validation': 'spatial-block CV, 3x5, identical to runs 1 and 2',
         'never_features': ['Weight_*', 'Year_region/Year/Region'],
         'neighbourhood_features': 'disease rates among TRAINING locations outside the query location\'s own spatial block: Gaussian kernels 50/150/400 km (run 2) and, new in run 3, the k = 5/15/40 nearest such locations; prior-shrunk',
         'new_learners_run3': ['GAM [E]', 'GP-kriging [E] (location-level Gaussian process on empirical logits, Matern 1.5 ARD + white noise)', 'LR [ELK]', 'ExtraTrees [ELK]', 'CatBoost-village [ELK]'],
         'proposed': PROPOSED,
         'fitness': f'scores rank-scaled by each model\'s training OOF ECDF; mean inner spatial-block ROC-AUC (average ranks for ties) minus {LAM_SD} x SD across blocks minus 0.001 x share of models selected',
         'bagging': f'{N_BAG} bootstrap resamples of training locations; weight vectors averaged; the same resamples are used for every optimiser',
         'equal_budget': f'{POP * ITERS} fitness evaluations per optimiser run; RB-PSO, RB-GWO, RB-DE and RB-HGWOPSO all use {N_BAG} bags',
         'success_rule_A': 'proposed mean spatial ROC-AUC is higher than EVERY single model (all 21) for each disease',
         'success_rule_B': 'pooled spatial ROC-AUC difference vs simple average has a 95% location-bootstrap CI above 0 for >= 2 diseases',
         'success_rule_C': 'proposed beats best single (inner-selected) with corrected t-test p < 0.05 for >= 2 diseases',
         'reporting': 'all methods, variants and ablations listed here are reported; none were added after seeing test results',
         'loro_regions': LORO_REGIONS, 'seed': SEED}
json.dump(RULES, open(OUT / f'decision_log_{VTAG}.json', 'w'), indent=1); RULES

# %% [markdown]
# ## 3. Splits (same as run 1)

# %%
SPLITS = []
def add(scheme, rep, fold, mask, block_col):
    SPLITS.append(dict(scheme=scheme, rep=rep, fold=str(fold), test=np.where(mask)[0], train=np.where(~mask)[0], block=block_col))
for f in range(K): add('random', 1, f, df.random_fold.values == f, 'block_r1')
for r in range(1, N_REP + 1):
    for f in range(K): add('spatial', r, f, df[f'spatial_fold_r{r}'].values == f, f'block_r{r}')
for reg in LORO_REGIONS: add('loro', 1, reg, df.Region.values == reg, 'block_r1')
if FAST: SPLITS = [s for sc in ('random', 'spatial', 'loro') for s in [x for x in SPLITS if x['scheme'] == sc][:2]]
RATIO = pd.DataFrame([{'s': s['scheme'], 'r': len(s['test']) / len(s['train'])} for s in SPLITS]).groupby('s').r.mean().to_dict()
print(len(SPLITS), 'splits:', pd.Series([s['scheme'] for s in SPLITS]).value_counts().to_dict())

# %% [markdown]
# ## 4. Features
# **Engineered (no labels used):** sea-level-equivalent temperature (separates climate from altitude, which were r = −0.96), De Martonne aridity index, temperature × humidity (insect-vector suitability), log rainfall, wind × humidity, latitude, longitude and two rotated coordinates.
#
# **Neighbourhood disease rates (labels used, training only):** for a location $x$, disease $d$ and bandwidth $b$:
# $$\text{lag}_{d,b}(x)=\frac{\sum_{j\in\text{train},\,\text{block}(j)\ne\text{block}(x)} K_b(x,j)\,y_{dj} + p_d}{\sum_{j} K_b(x,j) + 1},\qquad K_b=\exp\!\left(-\tfrac{\text{dist}^2}{2b^2}\right)$$
# Households in the location's own spatial block are always excluded, for training **and** test rows, so the feature means "what the surrounding region looks like" and never contains the household's own village. All three diseases' rates are offered to every disease model (a cross-disease signal).
#
# **New in run 3 — adaptive (k-nearest-location) rates.** A fixed 50 km kernel is often empty where villages are sparse, so it falls back to the national rate. We also compute, for each location, the disease rate among its **k = 5, 15 and 40 nearest training locations outside its own spatial block** (same prior shrinkage), plus the log-distance to the 15th such location. Feature set `ELK` = `EL` + these 10 features.
#
# **Kriging inputs.** The Gaussian-process learner uses the 10 climate variables (standardised) and projected coordinates in 100-km units.

# %%
lat0 = np.radians(df.Latitude.mean())
E = pd.DataFrame({
    'SL_temperature': df.Temperature + 0.0065 * df.Altitude,
    'DeMartonne': df.Precipitation / (df.Temperature + 10),
    'Temp_x_RH': df.Temperature * df.Relative_humidity / 100,
    'log_Precip': np.log1p(df.Precipitation),
    'Wind_x_RH': df.Wind_speed * df.Relative_humidity / 100,
    'Lat': df.Latitude, 'Lon': df.Longitude, 'LatPlusLon': df.Latitude + df.Longitude, 'LatMinusLon': df.Latitude - df.Longitude})
X_B = df[ENV].values.astype(float)
X_E = np.column_stack([X_B, E.values.astype(float)])
NAMES_B, NAMES_E = ENV, ENV + list(E.columns)

locdf = df.groupby('loc')[['Latitude', 'Longitude']].first().sort_index()
NL = len(locdf)
la, lo = np.radians(locdf.Latitude.values), np.radians(locdf.Longitude.values)
h = np.sin((la[None] - la[:, None]) / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None]) * np.sin((lo[None] - lo[:, None]) / 2) ** 2
DLOC = 2 * 6371 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
BWS = [50.0, 150.0, 400.0]
KERN = {b: np.exp(-DLOC ** 2 / (2 * b * b)) for b in BWS}
LOC = df['loc'].values
BLOCK_LOC = {c: df.groupby('loc')[c].first().sort_index().to_numpy(dtype=str) for c in df.columns if c.startswith('block_r')}
NAMES_LAG = [f'nbr_{d}_{int(b)}km' for b in BWS for d in DIS] + [f'nbr_density_{int(b)}km' for b in BWS]

def lag_features(fit_idx, pred_idx, block_col):
    """Neighbourhood disease rates from fit rows only, excluding the query location's own spatial block."""
    bl = BLOCK_LOC[block_col]; same = bl[:, None] == bl[None, :]
    n = np.bincount(LOC[fit_idx], minlength=NL).astype(float)
    pos = np.column_stack([np.bincount(LOC[fit_idx], weights=Y_ALL[fit_idx, k], minlength=NL) for k in range(3)])
    p0 = Y_ALL[fit_idx].mean(0)
    feats = []
    for b in BWS:
        Km = np.where(same, 0.0, KERN[b])
        num, den = Km @ pos, Km @ n
        feats.append((num + p0[None, :]) / (den[:, None] + 1.0))
    dens = np.column_stack([np.log1p(np.where(same, 0.0, KERN[b]) @ n) for b in BWS])
    L = np.column_stack(feats + [dens])            # per location
    return L[LOC[fit_idx]], L[LOC[pred_idx]]

KS = [5, 15, 40]
NAMES_KNN = [f'knn{k}_{d}' for k in KS for d in DIS] + ['log_dist_15th_location']

def knn_features(fit_idx, pred_idx, block_col):
    """Disease rates among the k nearest fit locations outside the query location's own spatial block (prior-shrunk)."""
    bl = BLOCK_LOC[block_col]; same = bl[:, None] == bl[None, :]
    n = np.bincount(LOC[fit_idx], minlength=NL).astype(float)
    pos = np.column_stack([np.bincount(LOC[fit_idx], weights=Y_ALL[fit_idx, k], minlength=NL) for k in range(3)])
    p0 = Y_ALL[fit_idx].mean(0)
    Dm = np.where(same | (n[None, :] == 0), np.inf, DLOC)
    order = np.argsort(Dm, axis=1, kind='stable'); navail = np.isfinite(Dm).sum(1); rr = np.arange(NL)
    cn = np.cumsum(n[order], axis=1); cp = np.cumsum(pos[order], axis=1); Ds = np.take_along_axis(Dm, order, axis=1)
    feats = []
    for k in KS:
        j = np.clip(np.minimum(k, navail) - 1, 0, NL - 1)
        feats.append(np.where((navail > 0)[:, None], (cp[rr, j] + p0[None, :]) / (cn[rr, j][:, None] + 1.0), p0[None, :]))
    j15 = np.clip(np.minimum(15, navail) - 1, 0, NL - 1)
    dist = np.where(navail > 0, Ds[rr, j15], DLOC.max())
    Kf = np.column_stack(feats + [np.log1p(dist)])
    return Kf[LOC[fit_idx]], Kf[LOC[pred_idx]]

# Gaussian-process (kriging) inputs: 10 climate variables + projected coordinates (100 km units)
XKM = df.Longitude.values * 111.32 * np.cos(lat0) / 100; YKM = df.Latitude.values * 110.57 / 100
X_G = np.column_stack([X_E[:, :10], XKM, YKM])

def featureset(name, fit_idx, pred_idx, block_col):
    if name == 'B': return X_B[fit_idx], X_B[pred_idx]
    if name in ('E', 'G'): return X_E[fit_idx], X_E[pred_idx]
    Lf, Lp = lag_features(fit_idx, pred_idx, block_col)
    if name == 'EL': return np.column_stack([X_E[fit_idx], Lf]), np.column_stack([X_E[pred_idx], Lp])
    Kf, Kp = knn_features(fit_idx, pred_idx, block_col)
    return np.column_stack([X_E[fit_idx], Lf, Kf]), np.column_stack([X_E[pred_idx], Lp, Kp])
print('Feature sets: B =', len(NAMES_B), ' E =', len(NAMES_E), ' EL =', len(NAMES_E) + len(NAMES_LAG), ' ELK =', len(NAMES_E) + len(NAMES_LAG) + len(NAMES_KNN))

# %% [markdown]
# ## 5. The model library
# Each entry is a model type × feature set. Hyperparameters are fixed in advance (identical in every split). `[B]` = 5 original variables (run-1 setting), `[E]` = + engineered, `[EL]` = + neighbourhood rates, `[ELK]` = + adaptive nearest-location rates. The first 16 learners are exactly the run-2 library; the last 5 are new.
#
# **GP-kriging [E]** works at location level: for each training location it takes the empirical logit $z=\log\frac{k+0.5}{n-k+0.5}$ of its $k$ positives among $n$ households, with known sampling variance $v=\frac1{k+0.5}+\frac1{n-k+0.5}$, and fits a Gaussian process $z = \mu + f(\text{climate},\text{space}) + \varepsilon$ with a Matérn-3/2 ARD kernel (length scales bounded below so the surface stays smooth) plus white noise. Predictions are the posterior mean, back-transformed with the logistic function.

# %%
def cb(**kw): return CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05, verbose=0, thread_count=1, random_seed=SEED, allow_writing_files=False, **kw)
LIB = [  # (name, kind, featureset)
    ('GAM [B]', 'single', 'B'), ('RF [B]', 'single', 'B'), ('XGB [B]', 'single', 'B'), ('CatBoost [B]', 'single', 'B'),
    ('CatBoost [E]', 'single', 'E'), ('kNN [E]', 'single', 'E'),
    ('LR [EL]', 'single', 'EL'), ('GAM [EL]', 'single', 'EL'), ('RF [EL]', 'single', 'EL'), ('ExtraTrees [EL]', 'single', 'EL'),
    ('XGB [EL]', 'single', 'EL'), ('CatBoost [EL]', 'single', 'EL'), ('MLP [EL]', 'single', 'EL'),
    ('CatBoost-groupbalanced [EL]', 'gb', 'EL'), ('CatBoost-village [EL]', 'loc', 'EL'), ('CatBoost-multitask [EL]', 'multi', 'EL'),
    # ---- new in run 3 ----
    ('GAM [E]', 'single', 'E'), ('GP-kriging [E]', 'gp', 'G'),
    ('LR [ELK]', 'single', 'ELK'), ('ExtraTrees [ELK]', 'single', 'ELK'), ('CatBoost-village [ELK]', 'loc', 'ELK')]
NAMES = [m[0] for m in LIB]; C = len(LIB)
IDX_RUN2 = list(range(16)); NEW = NAMES[16:]

def make_single(name):
    t = name.split(' [')[0]
    if t == 'LR': return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
    if t == 'GAM': return make_pipeline(StandardScaler(), SplineTransformer(n_knots=5, degree=3), LogisticRegression(C=0.5, max_iter=3000))
    if t == 'RF': return RandomForestClassifier(n_estimators=200, min_samples_leaf=10, max_features='sqrt', n_jobs=1, random_state=SEED)
    if t == 'ExtraTrees': return ExtraTreesClassifier(n_estimators=200, min_samples_leaf=10, max_features='sqrt', n_jobs=1, random_state=SEED)
    if t == 'XGB': return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, n_jobs=1, random_state=SEED, verbosity=0)
    if t == 'CatBoost': return cb()
    if t == 'kNN': return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=50))
    if t == 'MLP': return make_pipeline(StandardScaler(), MLPClassifier((32, 16), alpha=1e-2, max_iter=400, early_stopping=False, random_state=SEED))
    raise ValueError(name)

def gp_kriging(fit_idx, pred_idx, d):
    """Location-level Gaussian-process regression on empirical logits (kriging with climate covariates)."""
    lf = LOC[fit_idx]; g = pd.DataFrame(X_G[fit_idx]).groupby(lf).mean()
    n = pd.Series(lf).value_counts().reindex(g.index).values.astype(float)
    k = pd.Series(Y_ALL[fit_idx, d]).groupby(lf).sum().reindex(g.index).values.astype(float)
    z = np.log((k + 0.5) / (n - k + 0.5)); v = 1 / (k + 0.5) + 1 / (n - k + 0.5); mu = np.average(z, weights=1 / v)
    sc = StandardScaler().fit(g.values[:, :10])
    Xg = np.column_stack([sc.transform(g.values[:, :10]), g.values[:, 10:]]); Xp = np.column_stack([sc.transform(X_G[pred_idx, :10]), X_G[pred_idx, 10:]])
    kern = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=np.r_[np.full(10, 3.0), np.full(2, 2.0)], length_scale_bounds=(1.0, 1e5), nu=1.5) + WhiteKernel(0.1, (1e-3, 10.0))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gp = GaussianProcessRegressor(kern, alpha=v, normalize_y=False, n_restarts_optimizer=0, random_state=SEED).fit(Xg, z - mu)
        eta = gp.predict(Xp) + mu
    except Exception:
        eta = np.full(len(pred_idx), mu)
    return 1 / (1 + np.exp(-eta))

GROUP = df.Year_region.values
def fit_predict_lib(fit_idx, pred_idx, block_col):
    """Return (n_pred, 3 diseases, C models) probabilities."""
    out = np.zeros((len(pred_idx), 3, C)); cache = {}
    for j, (name, kind, fs) in enumerate(LIB):
        if kind == 'gp':
            for d in range(3): out[:, d, j] = gp_kriging(fit_idx, pred_idx, d)
            continue
        if fs not in cache: cache[fs] = featureset(fs, fit_idx, pred_idx, block_col)
        Xf, Xp = cache[fs]; Yf = Y_ALL[fit_idx]
        if kind == 'single':
            for d in range(3): out[:, d, j] = make_single(name).fit(Xf, Yf[:, d]).predict_proba(Xp)[:, 1]
        elif kind == 'gb':
            g = pd.Series(GROUP[fit_idx]); w = (1.0 / g.map(g.value_counts())).to_numpy(dtype=float, copy=True); w = w * len(w) / w.sum()
            for d in range(3): out[:, d, j] = cb().fit(Xf, Yf[:, d], sample_weight=w).predict_proba(Xp)[:, 1]
        elif kind == 'loc':
            lf = LOC[fit_idx]; agg = pd.DataFrame(Xf).groupby(lf).mean(); nloc = pd.Series(lf).value_counts().reindex(agg.index).values
            for d in range(3):
                yl = pd.Series(Yf[:, d]).groupby(lf).mean().reindex(agg.index).values
                reg = CatBoostRegressor(iterations=300, depth=4, learning_rate=0.05, verbose=0, thread_count=1, random_seed=SEED, allow_writing_files=False)
                out[:, d, j] = np.clip(reg.fit(agg.values, yl, sample_weight=nloc).predict(Xp), 1e-4, 1 - 1e-4)
        elif kind == 'multi':
            m = cb(loss_function='MultiLogloss').fit(Xf, Yf); out[:, :, j] = np.asarray(m.predict_proba(Xp))
    return out

def balanced_assign(groups, sizes, k, rng):
    load = np.zeros(k); out = {}
    for g in rng.permutation(np.array(groups, dtype=object)):
        f = int(np.argmin(load)); out[g] = f; load[f] += sizes[g]
    return out

def inner_folds(tr, block_col, seed, kind):
    if kind == 'spatial':
        b = df[block_col].values[tr]; sizes = pd.Series(b).value_counts().to_dict()
        asg = balanced_assign(list(sizes), sizes, K, np.random.default_rng(seed)); return np.array([asg[x] for x in b])
    return np.random.default_rng(seed).permutation(np.arange(len(tr)) % K)

def run_split(i):
    """Outer test predictions + inner spatial-block OOF predictions (the inner folds are identical to run 2)."""
    warnings.filterwarnings('ignore')
    s = SPLITS[i]; tr, te = s['train'], s['test']; t0 = time.time()
    res = {'i': i, 'test': fit_predict_lib(tr, te, s['block'])}
    fo = inner_folds(tr, s['block'], SEED + i, 'spatial'); P = np.zeros((len(tr), 3, C))
    for k in range(K):
        q = fo == k; P[q] = fit_predict_lib(tr[~q], tr[q], s['block'])
    res['oof_spatial'] = P; res['fold_spatial'] = fo
    res['seconds'] = time.time() - t0
    return res

t0 = time.time()
RES = sorted(Parallel(n_jobs=N_JOBS)(delayed(run_split)(i) for i in range(len(SPLITS))), key=lambda r: r['i'])
print(f'{len(SPLITS)} splits x {C} models x 3 diseases, with spatial inner CV: {(time.time() - t0) / 60:.1f} min')
Path('cache').mkdir(exist_ok=True)   # library predictions, kept outside the results folder for later re-use
np.savez_compressed(f'cache/library_predictions_{VTAG}.npz', **{f'{k}_{r["i"]}': r[k].astype(np.float32) for r in RES for k in ('test', 'oof_spatial')},
                    **{f'fold_{r["i"]}': r['fold_spatial'] for r in RES})

# %% [markdown]
# ## 6. How good is each single model? (effect of the new features)

# %%
def is_scored(scheme, fold, d): return scheme != 'loro' or elig[(fold, d)]
def metrics(y, p):
    two = len(np.unique(y)) > 1
    return {'ROC_AUC': roc_auc_score(y, p) if two else np.nan, 'PR_AUC': average_precision_score(y, p) if two else np.nan, 'Brier': brier_score_loss(y, p)}
rows = []
for r in RES:
    s = SPLITS[r['i']]
    for di, d in enumerate(DIS):
        y = Y_ALL[s['test'], di]
        for j, nm in enumerate(NAMES):
            rows.append({'scheme': s['scheme'], 'rep': s['rep'], 'fold': s['fold'], 'disease': d, 'model': nm, 'scored': is_scored(s['scheme'], s['fold'], d), **metrics(y, r['test'][:, di, j])})
SM = save_table(pd.DataFrame(rows), 'single_model_fold_metrics')
single_sum = SM[SM.scored].groupby(['scheme', 'disease', 'model']).ROC_AUC.mean().unstack(['scheme', 'disease'])
single_sum = single_sum[[(s, d) for s in ['random', 'spatial', 'loro'] for d in DIS]].reindex(NAMES)
save_table(single_sum.round(4), 'single_model_roc_auc_summary', index=True)
single_sum.round(3)

# %%
def fs_colour(n): return '#c3c2b7' if '[B]' in n else '#86b6ef' if '[E]' in n else '#2a78d6' if '[EL]' in n else '#0d3f86'
fig, axs = plt.subplots(1, 3, figsize=(14, 6.4), sharey=True)
for ax, d in zip(axs, DIS):
    v = single_sum[('spatial', d)].reindex(NAMES)
    ax.barh(range(C), v.values, color=[fs_colour(n) for n in NAMES], height=0.65); ax.axvline(0.5, ls='--', color='#e34948', lw=1)
    ax.set_xlim(0.45, max(0.8, np.nanmax(v.values) + 0.03)); ax.set_title(d, loc='left'); ax.set_xlabel('Mean ROC-AUC, spatial blocks')
axs[0].set_yticks(range(C)); axs[0].set_yticklabels(NAMES, fontsize=8.5); axs[0].invert_yaxis()
fig.suptitle('Single models: grey = 5 original variables, light blue = + engineered, blue = + neighbourhood rates, navy = + nearest-location rates', x=0.01, ha='left')
fig.tight_layout(); save_fig(fig, 'single_models_spatial_roc_auc')

# %% [markdown]
# ## 7. Proposed method: RB-HGWOPSO — robust, bagged hybrid GWO–PSO ensemble selection
# **Encoding (as in run 2).** An agent is a vector $x\in[-1,1]^{C}$, one entry per model. Model $m$ is selected if $x_m>0$; its weight is $x_m/\sum_{k:x_k>0}x_k$.
#
# **Step 1 — common rank scale.** Each model's scores are replaced by their position in that model's own training out-of-fold score distribution: $u_m(s)=\widehat F_m^{\text{oof}}(s)$ (mid-rank ECDF). Test scores use the same training ECDF, so nothing from the test fold is used. A CatBoost probability of 0.3 and a GAM probability of 0.3 no longer mean different things when averaged.
#
# **Step 2 — block-robust fitness.** With inner spatial blocks $b=1..5$ and $\text{AUC}_b$ the ROC-AUC (average ranks for ties) of the weighted score in block $b$:
# $$F(x)=\overline{\text{AUC}_b}-\lambda\,\text{SD}_b(\text{AUC}_b)-0.001\,\frac{|S(x)|}{C},\qquad \lambda=0.5.$$
# This rewards weights that work in every held-out area, not weights that are excellent in one area and poor in another. Run 2 used the pooled AUC with ties broken by row order.
#
# **Step 3 — hybrid GWO–PSO search (unchanged from run 2).** PSO velocity pulled towards the three GWO leader-guided positions; 20 agents × 30 iterations.
#
# **Step 4 — bagging.** The search is repeated on $B=10$ bootstrap resamples of training **locations** (all households of a sampled location move together), and the final weights are the average of the 10 weight vectors. The same 10 resamples are used for RB-PSO, RB-GWO and RB-DE, so the optimisers are compared on identical problems.
#
# **Cost.** One fitness evaluation is $O(nC + n\log n)$; one bagged run is $B\times P\times T$ evaluations (6,000), about one second here.

# %%
def auc_rows(y, S):
    """ROC-AUC for each row of S (pop x n), ties broken by order — the run-2 fitness, kept to reproduce SA-HGWOPSO exactly."""
    ranks = np.argsort(np.argsort(S, axis=1, kind='stable'), axis=1) + 1
    n1 = y.sum(); n0 = len(y) - n1
    return (ranks[:, y == 1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)

def auc_avg(y, S):
    """ROC-AUC for each row of S with average ranks for ties (identical to sklearn's roc_auc_score)."""
    R = stats.rankdata(S, axis=1); n1 = y.sum(); n0 = len(y) - n1
    return (R[:, y == 1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)

def decode(X):
    X = np.atleast_2d(X); sel = X > 0
    none = ~sel.any(1)
    if none.any(): sel[none, np.argmax(X[none], 1)] = True
    W = np.where(sel, np.maximum(X, 1e-6), 0.0); return W / W.sum(1, keepdims=True), sel

def make_fitness(P, y, lam=0.001):
    """Run-2 fitness: pooled OOF ROC-AUC (ties by order) minus sparsity penalty."""
    def f(X):
        W, sel = decode(X); return auc_rows(y, W @ P.T) - lam * sel.sum(1) / P.shape[1]
    return f

def block_groups(y, fo):
    return [g for g in (np.where(fo == k)[0] for k in np.unique(fo)) if 0 < y[g].sum() < len(g)]

def robust_value(S, y, groups, lam_sd=LAM_SD):
    A = np.column_stack([auc_avg(y[g], S[:, g]) for g in groups]); return A.mean(1) - lam_sd * A.std(1)

def make_fitness_robust(P, y, fo, lam_sd=LAM_SD, lam=0.001):
    groups = block_groups(y, fo)
    def f(X):
        W, sel = decode(X); return robust_value(W @ P.T, y, groups, lam_sd) - lam * sel.sum(1) / P.shape[1]
    return f

def make_fitness_pooled(P, y, lam=0.001):
    """Ablation: pooled OOF ROC-AUC with correct ties (no block robustness)."""
    def f(X):
        W, sel = decode(X); return auc_avg(y, W @ P.T) - lam * sel.sum(1) / P.shape[1]
    return f

def ecdf_scale(Po, Pt):
    """Mid-rank ECDF of each model's training OOF scores, applied to OOF and test scores."""
    n = len(Po); Ue = np.empty_like(Po); Ut = np.empty_like(Pt)
    for m in range(Po.shape[1]):
        s = np.sort(Po[:, m])
        Ue[:, m] = (stats.rankdata(Po[:, m]) - 0.5) / n
        Ut[:, m] = (np.searchsorted(s, Pt[:, m], 'left') + np.searchsorted(s, Pt[:, m], 'right')) / (2 * n)
    return Ue, Ut

def pso(f, M, seed, pop=POP, it=ITERS):
    rg = np.random.default_rng(seed); X = rg.uniform(-1, 1, (pop, M)); V = np.zeros_like(X); fx = f(X)
    pb, pbf = X.copy(), fx.copy(); g = pb[pbf.argmax()].copy(); curve = [pbf.max()]
    for t in range(it - 1):
        w = 0.9 - 0.5 * t / max(1, it - 2); r1, r2 = rg.random((2, pop, M))
        V = np.clip(w * V + 1.5 * r1 * (pb - X) + 1.5 * r2 * (g - X), -0.5, 0.5); X = np.clip(X + V, -1, 1); fx = f(X)
        m = fx > pbf; pb[m], pbf[m] = X[m], fx[m]; g = pb[pbf.argmax()].copy(); curve.append(pbf.max())
    return g, curve

def gwo(f, M, seed, pop=POP, it=ITERS, hybrid=False):
    rg = np.random.default_rng(seed); X = rg.uniform(-1, 1, (pop, M)); V = np.zeros_like(X); fx = f(X)
    o = np.argsort(-fx)[:3]; L, Lf = X[o].copy(), fx[o].copy(); curve = [Lf[0]]
    for t in range(it - 1):
        a = 2 - 2 * t / max(1, it - 2); Xn = []
        for j in range(3):
            r1, r2 = rg.random((2, pop, M)); A = 2 * a * r1 - a; Cc = 2 * r2; Xn.append(L[j] - A * np.abs(Cc * L[j] - X))
        if hybrid:
            w = 0.5 + rg.random() / 2; r = rg.random((3, pop, M))
            V = np.clip(w * (V + 0.5 * r[0] * (Xn[0] - X) + 0.5 * r[1] * (Xn[1] - X) + 0.5 * r[2] * (Xn[2] - X)), -0.5, 0.5); X = np.clip(X + V, -1, 1)
        else:
            X = np.clip((Xn[0] + Xn[1] + Xn[2]) / 3, -1, 1)
        fx = f(X); allX = np.vstack([L, X]); allf = np.concatenate([Lf, fx]); o = np.argsort(-allf)[:3]; L, Lf = allX[o].copy(), allf[o].copy(); curve.append(Lf[0])
    return L[0], curve

def de(f, M, seed, pop=POP, it=ITERS, F=0.6, CR=0.9):
    rg = np.random.default_rng(seed); X = rg.uniform(-1, 1, (pop, M)); fx = f(X); curve = [fx.max()]
    for t in range(it - 1):
        idx = np.array([rg.choice(np.delete(np.arange(pop), i), 3, replace=False) for i in range(pop)])
        Vm = np.clip(X[idx[:, 0]] + F * (X[idx[:, 1]] - X[idx[:, 2]]), -1, 1)
        cross = rg.random((pop, M)) < CR; cross[np.arange(pop), rg.integers(0, M, pop)] = True
        U = np.where(cross, Vm, X); fu = f(U); m = fu > fx; X[m], fx[m] = U[m], fu[m]; curve.append(fx.max())
    return X[fx.argmax()], curve

OPTS = {'PSO': lambda f, M, s: pso(f, M, s), 'GWO': lambda f, M, s: gwo(f, M, s), 'DE': lambda f, M, s: de(f, M, s),
        'HGWOPSO': lambda f, M, s: gwo(f, M, s, hybrid=True)}

def bags_for(lc, seed, B=N_BAG):
    """B bootstrap resamples of training locations (row indices; households of a location move together)."""
    rg = np.random.default_rng(seed); ul, inv = np.unique(lc, return_inverse=True)
    rows_of = np.split(np.argsort(inv, kind='stable'), np.cumsum(np.bincount(inv))[:-1])
    return [np.concatenate([rows_of[k] for k in rg.integers(0, len(ul), len(ul))]) for _ in range(B)]

def bagged(opt, P, y, fo, bags, robust=True):
    """Run an optimiser on every bag; return the averaged weight vector, curves and mean number selected."""
    Ws, curves, nsel = [], [], []
    for b, rows in enumerate(bags):
        f = make_fitness_robust(P[rows], y[rows], fo[rows]) if robust else make_fitness_pooled(P[rows], y[rows])
        x, cv = opt(f, P.shape[1], b); W, sel = decode(x); Ws.append(W[0]); curves.append(cv); nsel.append(int(sel.sum()))
    return np.mean(Ws, 0), curves, float(np.mean(nsel))

def caruana(P, y, iters=30):
    s = auc_avg(y, P.T); chosen = [int(np.argmax(s))]
    for _ in range(iters - 1):
        cur = P[:, chosen].sum(1); cand = (cur[None, :] + P.T) / (len(chosen) + 1)
        chosen.append(int(np.argmax(auc_avg(y, cand))))
    w = np.bincount(chosen, minlength=P.shape[1]).astype(float); return w / w.sum()

def logit(P): P = np.clip(P, 1e-4, 1 - 1e-4); return np.log(P / (1 - P))

RB = {'PSO': 'RB-PSO weights', 'GWO': 'RB-GWO weights', 'DE': 'RB-DE weights', 'HGWOPSO': PROPOSED}

def ensemble_split(r):
    warnings.filterwarnings('ignore')
    s = SPLITS[r['i']]; tr, te = s['train'], s['test']; out = []; curves = []
    fo = r['fold_spatial']; lc = LOC[tr]
    for di, d in enumerate(DIS):
        y = Y_ALL[tr, di]; yt = Y_ALL[te, di]; Po, Pt = r['oof_spatial'][:, di, :], r['test'][:, di, :]
        Pe, Pte = ecdf_scale(Po, Pt); groups = block_groups(y, fo); bags = bags_for(lc, SEED + 1000 * r['i'] + di)
        preds = {}
        inner_mean = np.column_stack([auc_avg(y[g], Po[g].T) for g in groups]).mean(1)
        preds['Best single (inner-selected)'] = Pt[:, int(np.argmax(inner_mean))]
        preds['Simple average (all models)'] = Pt.mean(1)
        preds['Simple average of ranks (all models)'] = Pte.mean(1)
        preds['Caruana greedy selection'] = Pt @ caruana(Po, y)
        preds['Stacking (logistic)'] = LogisticRegression(C=1.0, max_iter=3000).fit(logit(Po), y).predict_proba(logit(Pt))[:, 1]
        for sd in SEEDS:   # exact run-2 method, now on the 21-model library
            x, _ = gwo(make_fitness(Po, y), C, sd, hybrid=True); preds[('SA-HGWOPSO (run-2 method)', sd)] = Pt @ decode(x)[0][0]
        for name, opt in OPTS.items():   # robust bagged versions, identical bags
            t0 = time.time(); W, cvs, ns = bagged(opt, Pe, y, fo, bags); label = RB[name]
            preds[label] = Pte @ W
            out.append({'disease': d, 'method': label, 'seed': 0, 'seconds': time.time() - t0, 'inner_fitness': float(robust_value((Pe @ W)[None, :], y, groups)[0]),
                        'n_selected': ns, 'n_nonzero_final': int((W > 1e-3).sum()), **{f'w_{NAMES[j]}': W[j] for j in range(C)}})
            for b, cv in enumerate(cvs): curves.append({'disease': d, 'optimiser': name, 'seed': b, 'curve': cv})
        # ablations of the proposed method (each removes one component)
        x, _ = gwo(make_fitness_robust(Pe, y, fo), C, 0, hybrid=True); preds['Ablation: no bagging'] = Pte @ decode(x)[0][0]
        W, _, _ = bagged(OPTS['HGWOPSO'], Pe, y, fo, bags, robust=False); preds['Ablation: no robust fitness'] = Pte @ W
        W, _, _ = bagged(OPTS['HGWOPSO'], Po, y, fo, bags); preds['Ablation: no rank scaling'] = Pt @ W
        W, _, _ = bagged(OPTS['HGWOPSO'], Pe[:, IDX_RUN2], y, fo, bags); preds['Ablation: run-2 library only'] = Pte[:, IDX_RUN2] @ W
        for key, p in preds.items():
            meth, sd = (key if isinstance(key, tuple) else (key, 0))
            out.append({'disease': d, 'method': meth, 'seed': sd, 'pred': p, 'y': yt})
    return {'i': r['i'], 'rows': out, 'curves': curves}

t0 = time.time()
ENSR = Parallel(n_jobs=N_JOBS)(delayed(ensemble_split)(r) for r in RES)
print(f'Ensembles and optimisers done in {(time.time() - t0) / 60:.1f} min')

# %%
erows, oprows, crow, prow = [], [], [], []
for e in ENSR:
    s = SPLITS[e['i']]
    for o in e['rows']:
        if 'pred' in o:
            erows.append({'scheme': s['scheme'], 'rep': s['rep'], 'fold': s['fold'], 'disease': o['disease'], 'method': o['method'], 'seed': o['seed'],
                          'scored': is_scored(s['scheme'], s['fold'], o['disease']), **metrics(o['y'], o['pred'])})
            if o['seed'] == 0: prow.append(pd.DataFrame({'scheme': s['scheme'], 'rep': s['rep'], 'fold': s['fold'], 'idx': s['test'], 'disease': o['disease'], 'method': o['method'], 'p': o['pred'], 'y': o['y']}))
        else:
            oprows.append({'scheme': s['scheme'], 'rep': s['rep'], 'fold': s['fold'], **{k: v for k, v in o.items()}})
    for c in e['curves']: crow.append({'scheme': s['scheme'], **c})
EM = pd.DataFrame(erows); EMs = EM.groupby(['scheme', 'rep', 'fold', 'disease', 'method', 'scored'])[['ROC_AUC', 'PR_AUC', 'Brier']].mean().reset_index()
EM_seed_sd = EM.groupby(['scheme', 'rep', 'fold', 'disease', 'method']).ROC_AUC.std().groupby(['scheme', 'disease', 'method']).mean()
OPR = pd.DataFrame(oprows); EPRED = pd.concat(prow, ignore_index=True)
EPRED['ID'] = df.ID.values[EPRED.idx.values]; EPRED['Location_id'] = df.Location_id.values[EPRED.idx.values]
save_table(EMs, 'ensemble_fold_metrics'); save_table(OPR, 'optimiser_runs'); save_table(EPRED.drop(columns='idx'), 'ensemble_test_predictions')
METHODS = ['Best single (inner-selected)', 'Simple average (all models)', 'Simple average of ranks (all models)', 'Caruana greedy selection',
           'Stacking (logistic)', 'SA-HGWOPSO (run-2 method)', 'RB-PSO weights', 'RB-GWO weights', 'RB-DE weights', PROPOSED]
ABL = ['Ablation: no bagging', 'Ablation: no robust fitness', 'Ablation: no rank scaling', 'Ablation: run-2 library only']
ens_sum = EMs[EMs.scored].groupby(['scheme', 'disease', 'method'])[['ROC_AUC', 'PR_AUC', 'Brier']].mean()
tab = ens_sum.ROC_AUC.unstack(['scheme', 'disease'])[[(s, d) for s in ['random', 'spatial', 'loro'] for d in DIS]].reindex(METHODS + ABL)
save_table(tab.round(4), 'ensemble_roc_auc_summary', index=True)
tab.round(3)

# %% [markdown]
# ## 8. Does the proposed method beat every single model? (spatial blocks)
# Paired comparisons on the same 15 spatial test folds: corrected resampled t-test (Nadeau & Bengio 2003) and a 95% CI for the pooled ROC-AUC difference from resampling whole locations. The proposed method is compared with **all 21 single models** and with every other ensemble method and ablation.

# %%
loc_codes = df.Location_id.values
def corrected_t(d_, ratio):
    d_ = np.asarray(d_, float); J = len(d_); v = d_.var(ddof=1)
    if J < 2 or v == 0: return np.nan
    t = d_.mean() / np.sqrt((1 / J + ratio) * v); return 2 * stats.t.sf(abs(t), J - 1)
def fold_series(kind, name, d, scheme='spatial'):
    if kind == 'ens': s_ = EMs[(EMs.scheme == scheme) & (EMs.method == name) & (EMs.disease == d) & EMs.scored]
    else: s_ = SM[(SM.scheme == scheme) & (SM.model == name) & (SM.disease == d) & SM.scored]
    return s_.set_index(['rep', 'fold']).ROC_AUC.sort_index()
def pooled_matrix(kind, name, d):
    """Spatial-block test predictions as a table: household index x repeat."""
    if kind == 'ens':
        sub = EPRED[(EPRED.scheme == 'spatial') & (EPRED.method == name) & (EPRED.disease == d)]; return sub.pivot_table(index='idx', columns='rep', values='p').sort_index()
    j = NAMES.index(name); rows_ = []
    for r in RES:
        s_ = SPLITS[r['i']]
        if s_['scheme'] == 'spatial': rows_.append(pd.DataFrame({'idx': s_['test'], 'rep': s_['rep'], 'p': r['test'][:, DIS.index(d), j]}))
    return pd.concat(rows_).pivot_table(index='idx', columns='rep', values='p').sort_index()
def boot_diff(d, A, Bm, B=B_BOOT, seed=SEED):
    """Pooled ROC-AUC difference with a 95% CI from resampling whole locations."""
    common = A.index.intersection(Bm.index); A, Bm = A.loc[common], Bm.loc[common]
    y = Y_ALL[common.values, DIS.index(d)]; lc = loc_codes[common.values]
    groups = [np.where(lc == l)[0] for l in np.unique(lc)]; rg = np.random.default_rng(seed); vals = []
    auc_d = lambda ix: np.mean([roc_auc_score(y[ix], A[c].values[ix]) for c in A.columns]) - np.mean([roc_auc_score(y[ix], Bm[c].values[ix]) for c in Bm.columns])
    pt = auc_d(np.arange(len(y)))
    for _ in range(B):
        ix = np.concatenate([groups[k] for k in rg.integers(0, len(groups), len(groups))]); vals.append(auc_d(ix))
    return pt, *np.percentile(vals, [2.5, 97.5])

def compare_one(d, kind, name):
    a = fold_series('ens', PROPOSED, d); b = fold_series(kind, name, d)
    pt, lo, hi = boot_diff(d, pooled_matrix('ens', PROPOSED, d), pooled_matrix(kind, name, d), B=max(100, B_BOOT // 2))
    la = fold_series('ens', PROPOSED, d, 'loro'); lb = fold_series(kind, name, d, 'loro')
    return {'disease': d, 'versus_kind': 'single model' if kind == 'single' else ('ablation' if name in ABL else 'ensemble method'), 'versus': name,
            'proposed_ROC_AUC': a.mean(), 'other_ROC_AUC': b.mean(), 'dROC_AUC': (a - b).mean(), 'folds_better': f'{int(((a - b) > 0).sum())}/{len(a)}',
            'p_corrected': corrected_t(a - b, RATIO['spatial']), 'pooled_dAUC': pt, 'CI_lo': lo, 'CI_hi': hi,
            'LORO_dAUC': (la - lb).mean(), 'LORO_regions_better': f'{int(((la - lb) > 0).sum())}/{len(la)}'}
jobs = [(d, 'single', n) for d in DIS for n in NAMES] + [(d, 'ens', m) for d in DIS for m in METHODS + ABL if m != PROPOSED]
CMP = save_table(pd.DataFrame([compare_one(*j) for j in jobs]), 'proposed_vs_others')
CMP.sort_values(['disease', 'other_ROC_AUC'], ascending=[True, False]).groupby('disease').head(8).round(4)

# %%
fig, axs = plt.subplots(1, 3, figsize=(14, 6.4), sharey=False)
for ax, d in zip(axs, DIS):
    v = single_sum[('spatial', d)].sort_values(ascending=True); pv = float(ens_sum.loc[('spatial', d, PROPOSED), 'ROC_AUC'])
    av = float(ens_sum.loc[('spatial', d, 'Simple average (all models)'), 'ROC_AUC'])
    ax.barh(range(len(v)), v.values, color=[fs_colour(n) for n in v.index], height=0.65)
    ax.axvline(pv, color='#eb6834', lw=2, label=f'Proposed RB-HGWOPSO ({pv:.3f})'); ax.axvline(av, color='#55554f', lw=1.2, ls='--', label=f'Simple average ({av:.3f})')
    ax.set_yticks(range(len(v))); ax.set_yticklabels(v.index, fontsize=8); ax.set_xlim(0.45, max(0.8, v.max() + 0.03))
    ax.set_title(f'{d}: proposed beats {int((v < pv).sum())} of {len(v)} single models', loc='left', fontsize=10)
    ax.set_xlabel('Mean ROC-AUC, spatial blocks'); ax.legend(frameon=False, fontsize=7.5, loc='lower right')
fig.tight_layout(); save_fig(fig, 'proposed_vs_every_single_model')

# %% [markdown]
# ### Friedman tests with Nemenyi critical differences (15 spatial folds × 3 diseases = 45 blocks)
# (a) all ensemble methods; (b) the proposed method against all 21 single models.

# %%
def qalpha(k): return float(stats.studentized_range.ppf(0.95, k, 1e6) / np.sqrt(2))
def friedman_block(mat):
    fr = stats.friedmanchisquare(*[mat[c].values for c in mat.columns]); rk = mat.rank(axis=1, ascending=False).mean().sort_values()
    cd = qalpha(mat.shape[1]) * np.sqrt(mat.shape[1] * (mat.shape[1] + 1) / (6 * len(mat))); return fr, rk, cd
def rank_plot(rk, cd, title, name, height):
    fig, ax = plt.subplots(figsize=(8, height))
    ax.hlines(range(len(rk)), 1, rk.values, color='#e1e0d9')
    ax.scatter(rk.values, range(len(rk)), color=['#eb6834' if n == PROPOSED else '#2a78d6' for n in rk.index], zorder=3)
    ax.axvspan(rk.values[0], rk.values[0] + cd, color='#cde2fb', alpha=0.6, label=f'Not significantly different from the best (Nemenyi CD = {cd:.2f})')
    ax.set_yticks(range(len(rk))); ax.set_yticklabels(rk.index, fontsize=8.5); ax.invert_yaxis(); ax.set_xlabel('Mean rank (1 = best ROC-AUC)')
    ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(0.0, -0.12)); ax.set_title(title + ' (orange = proposed)', loc='left'); fig.tight_layout(); save_fig(fig, name)
mat_e = EMs[(EMs.scheme == 'spatial') & EMs.method.isin(METHODS)].pivot_table(index=['disease', 'rep', 'fold'], columns='method', values='ROC_AUC')[METHODS]
fr_e, rk_e, cd_e = friedman_block(mat_e)
mat_s = SM[(SM.scheme == 'spatial')].pivot_table(index=['disease', 'rep', 'fold'], columns='model', values='ROC_AUC')[NAMES]
mat_s[PROPOSED] = mat_e[PROPOSED]
fr_s, rk_s, cd_s = friedman_block(mat_s)
rank_tab = save_table(pd.concat([pd.DataFrame({'comparison': 'ensemble methods', 'mean_rank': rk_e, 'mean_ROC_AUC': mat_e.mean()[rk_e.index]}),
                                 pd.DataFrame({'comparison': 'proposed vs 21 single models', 'mean_rank': rk_s, 'mean_ROC_AUC': mat_s.mean()[rk_s.index]})]).rename_axis('method'), 'friedman_ranks', index=True)
print(f'(a) ensembles: Friedman p = {fr_e.pvalue:.2g}, CD = {cd_e:.2f}   (b) proposed vs single models: Friedman p = {fr_s.pvalue:.2g}, CD = {cd_s:.2f}')
rank_plot(rk_e, cd_e, 'Ensemble methods: mean rank over 45 spatial blocks', 'friedman_ranks_ensembles', 3.9)
rank_plot(rk_s, cd_s, 'Proposed vs all 21 single models: mean rank over 45 spatial blocks', 'friedman_ranks_vs_single_models', 6.2)
rank_tab.round(3)

# %% [markdown]
# ## 9. Optimisers inside the robust bagged scheme: fitness, test ROC-AUC and cost

# %%
CUR = pd.DataFrame(crow)
osum = (OPR[OPR.scheme == 'spatial'].groupby(['disease', 'method'])[['inner_fitness', 'seconds', 'n_selected', 'n_nonzero_final']].mean()
        .join(ens_sum.loc['spatial'][['ROC_AUC']]))
save_table(osum.reset_index(), 'optimiser_summary'); save_table(CUR.assign(curve=CUR.curve.map(lambda c: ' '.join(f'{v:.5f}' for v in c))), 'optimiser_convergence')
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
oc = {'PSO': '#898781', 'GWO': '#eb6834', 'DE': '#1baf7a', 'HGWOPSO': '#2a78d6'}
for ax, d in zip(axs, DIS):
    for o, c in oc.items():
        sub = CUR[(CUR.scheme == 'spatial') & (CUR.disease == d) & (CUR.optimiser == o)]
        arr = np.array(sub.curve.tolist()); arr = arr - arr[:, :1]
        ax.plot(range(1, arr.shape[1] + 1), arr.mean(0), color=c, label=o, lw=2 if o == 'HGWOPSO' else 1.3)
    ax.set_title(d, loc='left'); ax.set_xlabel('Iteration'); ax.grid(color='#eeeeea')
axs[0].set_ylabel('Robust fitness gain over iteration 1'); axs[2].legend(frameon=False, fontsize=8)
fig.suptitle('Convergence on bootstrap bags of the inner spatial folds (mean over bags and folds)', x=0.01, ha='left'); fig.tight_layout(); save_fig(fig, 'optimiser_convergence')
osum.round(4)

# %% [markdown]
# ## 10. Which models does the proposed method use?

# %%
wcols = [c for c in OPR.columns if c.startswith('w_')]
wmean = OPR[(OPR.method == PROPOSED) & (OPR.scheme == 'spatial')].groupby('disease')[wcols].mean().T
wmean.index = [c[2:] for c in wmean.index]; save_table(wmean.round(4), 'proposed_mean_weights', index=True)
fig, ax = plt.subplots(figsize=(8, 6.5))
xx = np.arange(len(wmean)); bw = 0.27
for k, d in enumerate(DIS): ax.barh(xx + (k - 1) * bw, wmean[d].values, height=bw, color=COL[d], label=d)
ax.set_yticks(xx); ax.set_yticklabels(wmean.index, fontsize=8.5); ax.invert_yaxis(); ax.set_xlabel('Mean ensemble weight (spatial folds)'); ax.legend(frameon=False)
ax.set_title('Weights chosen by the proposed RB-HGWOPSO ensemble', loc='left'); fig.tight_layout(); save_fig(fig, 'proposed_weights')
wmean.round(3)

# %% [markdown]
# ## 11. Run 2 method vs run 3 at a glance
# "Best single model (hindsight)" is the single model with the highest test score, chosen **after** seeing the test folds — an optimistic yardstick that cannot be used in practice.

# %%
sm_mean = SM[SM.scored].groupby(['scheme', 'disease', 'model']).ROC_AUC.mean()
best_any = sm_mean.groupby(['scheme', 'disease']).max(); best_name = sm_mean.groupby(['scheme', 'disease']).idxmax().map(lambda t: t[2])
gl = lambda m: ens_sum.ROC_AUC.xs(m, level='method')
glance = pd.DataFrame({'Run-2 method (SA-HGWOPSO)': gl('SA-HGWOPSO (run-2 method)'), 'Simple average': gl('Simple average (all models)'),
                       'Best single model (hindsight)': best_any, 'Proposed RB-HGWOPSO': gl(PROPOSED)}).reindex([(s, d) for s in ['random', 'spatial', 'loro'] for d in DIS])
glance['best_single_model_name'] = best_name.reindex(glance.index)
save_table(glance.round(4), 'run2_vs_run3', index=True)
fig, axs = plt.subplots(1, 3, figsize=(14, 3.9), sharey=True)
for ax, s in zip(axs, ['random', 'spatial', 'loro']):
    g = glance.loc[s]; xx = np.arange(3); bw = 0.2; cols4 = ['#c3c2b7', '#86b6ef', '#2a78d6', '#eb6834']
    for k, (c, colr) in enumerate(zip(glance.columns[:4], cols4)):
        ax.bar(xx + (k - 1.5) * bw, g[c].values, width=bw, color=colr, label=c)
        for xi, v in zip(xx + (k - 1.5) * bw, g[c].values): ax.text(xi, v + 0.005, f'{v:.2f}', ha='center', fontsize=6.8)
    ax.axhline(0.5, ls='--', color='#e34948', lw=1); ax.set_xticks(xx); ax.set_xticklabels(DIS); ax.set_ylim(0.4, 1.0)
    ax.set_title({'random': 'Random CV', 'spatial': 'Spatial blocks (primary)', 'loro': 'Leave-region-out'}[s], loc='left')
axs[0].set_ylabel('Mean ROC-AUC'); axs[0].legend(frameon=False, fontsize=7.5, loc='upper left')
fig.tight_layout(); save_fig(fig, 'run2_vs_run3')
glance.round(3)

# %% [markdown]
# ## 12. Summary against the pre-registered success rules

# %%
prop = gl(PROPOSED); sa = gl('Simple average (all models)')
ruleA = {d: bool(prop[('spatial', d)] > best_any[('spatial', d)]) for d in DIS}
cB = CMP[CMP.versus == 'Simple average (all models)'].set_index('disease'); ruleB = {d: bool(cB.loc[d, 'CI_lo'] > 0) for d in DIS}
cC = CMP[CMP.versus == 'Best single (inner-selected)'].set_index('disease'); ruleC = {d: bool(cC.loc[d, 'p_corrected'] < 0.05 and cC.loc[d, 'dROC_AUC'] > 0) for d in DIS}
SUMMARY = {'run_folder': str(OUT), 'versions': VERSIONS, 'n_models': C, 'new_models': NEW,
           'spatial_roc_auc': {d: {'proposed': round(float(prop[('spatial', d)]), 4), 'simple_average': round(float(sa[('spatial', d)]), 4),
                                   'run2_method': round(float(gl('SA-HGWOPSO (run-2 method)')[('spatial', d)]), 4),
                                   'best_single_hindsight': round(float(best_any[('spatial', d)]), 4), 'best_single_name': best_name[('spatial', d)],
                                   'single_models_beaten': int((single_sum[('spatial', d)] < prop[('spatial', d)]).sum())} for d in DIS},
           'random_roc_auc_proposed': {d: round(float(prop[('random', d)]), 4) for d in DIS},
           'loro_roc_auc_proposed': {d: round(float(prop[('loro', d)]), 4) for d in DIS},
           'friedman_ensembles': {'p': float(fr_e.pvalue), 'cd': cd_e, 'mean_ranks': rk_e.round(3).to_dict()},
           'friedman_vs_single_models': {'p': float(fr_s.pvalue), 'cd': cd_s, 'mean_ranks': rk_s.round(3).to_dict()},
           'rule_A_beats_every_single_model': ruleA, 'rule_A_met': all(ruleA.values()),
           'rule_B_ci_above_simple_average': ruleB, 'rule_B_met': sum(ruleB.values()) >= 2,
           'rule_C_beats_inner_selected_single_p05': ruleC, 'rule_C_met': sum(ruleC.values()) >= 2}
json.dump(SUMMARY, open(OUT / f'run_summary_{VTAG}.json', 'w'), indent=1)
print(json.dumps(SUMMARY['spatial_roc_auc'], indent=1))
print('Rule A (beats every single model):', ruleA, '\nRule B (CI above simple average):', ruleB, '\nRule C (p<0.05 vs inner-selected single):', ruleC)
print('\nFiles:', *sorted(p.name for p in OUT.rglob('*') if p.is_file()), sep='\n')
