"""Feature building from a decoded spec, learners (search surrogates and final model) and the spatially robust fitness."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from .features import CLIMATE_NAMES, COORD_NAMES

LAMBDA_SD, MU_COMPLEXITY, SEED = 0.5, 0.005, 20261119


def auc_avg(y, s):
    """ROC-AUC with average ranks for ties (identical to sklearn.metrics.roc_auc_score)."""
    y = np.asarray(y); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


class FeatureBuilder:
    def __init__(self, df, Y, NB, climate_coords_frame):
        self.C = climate_coords_frame; self.NB = NB; self.Y = Y; self.loc = df['loc'].values
        self.col = {c: i for i, c in enumerate(self.C.columns)}; self.Cv = self.C.values

    def build(self, spec, fit_idx, pred_idx, row_blocks, disease):
        cols = [self.col[c] for c in spec['climate']] + ([self.col[c] for c in COORD_NAMES] if spec['coordinates'] else [])
        Ff, Fp, names = [self.Cv[fit_idx][:, cols]], [self.Cv[pred_idx][:, cols]], [self.C.columns[c] for c in cols]
        if spec['kernel_rates'] or spec['knn_rates'] or spec['idw_rates']:
            dis = (0, 1, 2) if spec['cross_disease_rates'] else (disease,)
            a, b, n = self.NB.rates(fit_idx, pred_idx, row_blocks,
                                    bandwidths=spec['bandwidths'] if spec['kernel_rates'] else (),
                                    ks=tuple(spec.get('ks', (spec['k'],))) if spec['knn_rates'] else (), prior=spec['prior'],
                                    idw_power=spec['idw_power'] if spec['idw_rates'] else None,
                                    density=spec['density_terms'], diseases=dis)
            Ff.append(a); Fp.append(b); names += n
        return np.column_stack(Ff), np.column_stack(Fp), names


def fit_predict(kind, spec, Xf, yf, Xp, loc_fit, n_trees=None, seed=SEED):
    """kind: 'loc' (location-level ExtraTrees regressor, surrogate), 'row' (row-level ExtraTrees 100, surrogate),
    'full' (row-level ExtraTrees 300, final learner)."""
    msl, mf = spec['min_samples_leaf'], spec['max_features']
    if kind == 'loc':
        g = pd.DataFrame(Xf).groupby(loc_fit); Xl = g.mean().values
        nl = g.size().values.astype(float); yl = pd.Series(yf).groupby(loc_fit).mean().values
        m = ExtraTreesRegressor(n_trees or 100, min_samples_leaf=max(1, int(round(msl / 7.2))), max_features=mf, n_jobs=1, random_state=seed)
        return m.fit(Xl, yl, sample_weight=nl).predict(Xp)
    m = ExtraTreesClassifier(n_trees or (100 if kind == 'row' else 300), min_samples_leaf=msl, max_features=mf, n_jobs=1, random_state=seed)
    return m.fit(Xf, yf).predict_proba(Xp)[:, 1]


def robust_fitness(spec, FB, inner_pairs, row_blocks, disease, kind='loc', lam=LAMBDA_SD, mu=MU_COMPLEXITY, return_parts=False):
    """F = mean_b AUC_b - lam * SD_b AUC_b - mu * complexity, over the inner spatial folds."""
    aucs = []
    for fit, val in inner_pairs:
        Xf, Xp, _ = FB.build(spec, fit, val, row_blocks, disease)
        p = fit_predict(kind, spec, Xf, FB.Y[fit, disease], Xp, FB.loc[fit])
        aucs.append(auc_avg(FB.Y[val, disease], p))
    a = np.array(aucs, float); a = a[~np.isnan(a)]
    F = a.mean() - lam * a.std() - mu * spec['complexity']
    return (F, a) if return_parts else F
