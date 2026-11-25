"""Paper 1 (IJIES) analysis core, frozen 2026-11-21. Used by the experiment runner and by the notebook.

Corrections relative to the earlier pipeline (both documented in the feature-leakage audit, results/audit_v1):
  1. Deterministic kNN tie-break: candidate locations are ordered by (distance rounded to 1e-6 km, location ID).
     Coordinates are given to 0.1 degree, so many neighbours are exactly equidistant; the earlier order depended on
     floating-point noise (about 1e-12 km).
  2. Corrected fallback prior in the old 23-model library: p0 excludes the query location's own spatial block,
     as in the new code.
Everything else (feature definitions, learners, hyperparameters, folds) is unchanged from the frozen design.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import rankdata

SRC = Path(__file__).resolve().parents[1]; ROOT = SRC.parent
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from common.data import load, labels, ENV, SEED                     # noqa: E402
from common.features import Neighbourhood, climate_coords, CLIMATE_NAMES, COORD_NAMES  # noqa: E402
from common.fitness import FeatureBuilder                          # noqa: E402

DIS = ['CBPP', 'FMD', 'LSD']
OUT = ROOT / 'results' / 'paper1_v1'
TIE_DECIMALS = 6


# ----------------------------------------------------------------------------------------------- features
class NeighbourhoodV2(Neighbourhood):
    """Identical to common.features.Neighbourhood except for the deterministic kNN ordering."""

    def rates(self, fit_idx, pred_idx, row_blocks, bandwidths=(50, 150, 400), ks=(5, 15, 40), prior=1.0,
              idw_power=None, density=True, diseases=(0, 1, 2)):
        bl = self.block_of_loc(row_blocks); same = bl[:, None] == bl[None, :]
        n = np.bincount(self.loc[fit_idx], minlength=self.NL).astype(float)
        pos = np.column_stack([np.bincount(self.loc[fit_idx], weights=self.Y[fit_idx, k], minlength=self.NL) for k in range(3)])
        n_own = same.astype(float) @ n; pos_own = same.astype(float) @ pos
        p0 = (pos.sum(0)[None, :] - pos_own) / np.maximum(n.sum() - n_own, 1.0)[:, None]
        avail = (~same) & (n[None, :] > 0)
        cols, names = [], []
        for b in bandwidths:
            K = np.where(avail, np.exp(-self.D ** 2 / (2.0 * b * b)), 0.0)
            r = (K @ pos + prior * p0) / ((K @ n)[:, None] + prior)
            cols += [r[:, k] for k in diseases]; names += [f'kern{int(b)}_{k}' for k in diseases]
            if density: cols.append(np.log1p(K @ n)); names.append(f'dens{int(b)}')
        if len(ks):
            Dm = np.where(avail, self.D, np.inf)
            order = np.argsort(np.round(Dm, TIE_DECIMALS), axis=1, kind='stable')     # ties -> lower location ID first
            nav = np.isfinite(Dm).sum(1); rr = np.arange(self.NL)
            cn = np.cumsum(n[order], axis=1); cp = np.cumsum(pos[order], axis=1); Ds = np.take_along_axis(Dm, order, axis=1)
            for k_ in ks:
                j = np.clip(np.minimum(int(k_), nav) - 1, 0, self.NL - 1)
                r = np.where((nav > 0)[:, None], (cp[rr, j] + prior * p0) / (cn[rr, j][:, None] + prior), p0)
                cols += [r[:, k] for k in diseases]; names += [f'knn{int(k_)}_{k}' for k in diseases]
                if density: cols.append(np.log1p(np.where(nav > 0, Ds[rr, j], self.D.max()))); names.append(f'knndist{int(k_)}')
        if idw_power:
            W = np.where(avail & (self.D > 0), 1.0 / np.maximum(self.D, 1.0) ** idw_power, 0.0)
            r = (W @ pos + prior * p0) / ((W @ n)[:, None] + prior)
            cols += [r[:, k] for k in diseases]; names += [f'idw_{k}' for k in diseases]
        F = np.column_stack(cols)
        return F[self.loc[fit_idx]], F[self.loc[pred_idx]], names


_BASE = {'coordinates': False, 'kernel_rates': False, 'knn_rates': False, 'idw_rates': False, 'density_terms': False, 'cross_disease_rates': False,
         'bandwidths': (50.0, 150.0, 400.0), 'ks': (5, 15, 40), 'k': 15, 'prior': 1.0, 'idw_power': 2.0, 'min_samples_leaf': 10, 'max_features': 'sqrt', 'complexity': 0}
FEATURE_SETS = {'FS1': {**_BASE, 'climate': ENV},
                'FS5': {**_BASE, 'climate': CLIMATE_NAMES, 'coordinates': True, 'kernel_rates': True, 'knn_rates': True, 'density_terms': True, 'cross_disease_rates': True}}


def setup(Y_override=None):
    """Load data and build the feature machinery (optionally on permuted labels for negative controls)."""
    df = load(ROOT / 'data/raw'); Y = labels(df) if Y_override is None else Y_override
    NB = NeighbourhoodV2(df, Y); FB = FeatureBuilder(df, Y, NB, climate_coords(df))
    return df, Y, NB, FB


def dev_splits(df, reps=(1, 2, 3)):
    """Development outer splits: (rep, fold, train_idx, test_idx, block_column)."""
    out = []
    for r in reps:
        fo = df[f'spatial_fold_r{r}'].values
        for f in range(5): out.append((r, f, np.where(fo != f)[0], np.where(fo == f)[0], f'block_r{r}'))
    return out


def extratrees(seed=SEED):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(300, min_samples_leaf=10, max_features='sqrt', n_jobs=1, random_state=seed)


# ----------------------------------------------------------------------------------------------- old 23-model library, corrected
def load_library():
    """Run-3 library code (data/reference) with (i) the corrected fallback prior and (ii) deterministic kNN ties."""
    from audit_common_v1 import load_run3
    g = load_run3(scratch='_paper1_scratch')
    LOC, NL, Y_ALL, KERN, BWS, DLOC, KS = g['LOC'], g['NL'], g['Y_ALL'], g['KERN'], g['BWS'], g['DLOC'], g['KS']

    def p0_excl(fit_idx, same):
        n = np.bincount(LOC[fit_idx], minlength=NL).astype(float)
        pos = np.column_stack([np.bincount(LOC[fit_idx], weights=g['Y_ALL'][fit_idx, k], minlength=NL) for k in range(3)])
        S_ = same.astype(float); return n, pos, (pos.sum(0)[None, :] - S_ @ pos) / np.maximum(n.sum() - S_ @ n, 1.0)[:, None]

    def lag_features(fit_idx, pred_idx, block_col):
        bl = g['BLOCK_LOC'][block_col]; same = bl[:, None] == bl[None, :]; n, pos, p0 = p0_excl(fit_idx, same); feats = []
        for b in BWS:
            Km = np.where(same, 0.0, KERN[b]); feats.append((Km @ pos + p0) / ((Km @ n)[:, None] + 1.0))
        dens = np.column_stack([np.log1p(np.where(same, 0.0, KERN[b]) @ n) for b in BWS]); L = np.column_stack(feats + [dens])
        return L[LOC[fit_idx]], L[LOC[pred_idx]]

    def knn_features(fit_idx, pred_idx, block_col):
        bl = g['BLOCK_LOC'][block_col]; same = bl[:, None] == bl[None, :]; n, pos, p0 = p0_excl(fit_idx, same)
        Dm = np.where(same | (n[None, :] == 0), np.inf, DLOC); order = np.argsort(np.round(Dm, TIE_DECIMALS), axis=1, kind='stable')
        navail = np.isfinite(Dm).sum(1); rr = np.arange(NL)
        cn = np.cumsum(n[order], axis=1); cp = np.cumsum(pos[order], axis=1); Ds = np.take_along_axis(Dm, order, axis=1); feats = []
        for k in KS:
            j = np.clip(np.minimum(k, navail) - 1, 0, NL - 1)
            feats.append(np.where((navail > 0)[:, None], (cp[rr, j] + p0) / (cn[rr, j][:, None] + 1.0), p0))
        j15 = np.clip(np.minimum(15, navail) - 1, 0, NL - 1); dist = np.where(navail > 0, Ds[rr, j15], DLOC.max())
        Kf = np.column_stack(feats + [np.log1p(dist)]); return Kf[LOC[fit_idx]], Kf[LOC[pred_idx]]

    g['lag_features'] = lag_features; g['knn_features'] = knn_features
    return g


# ----------------------------------------------------------------------------------------------- metrics
def auc(y, s):
    y = np.asarray(y); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pr_auc(y, s):
    from sklearn.metrics import average_precision_score
    return average_precision_score(y, s) if 0 < np.sum(y) < len(y) else np.nan


def within_year_auc(y, s, year):
    """Pair-weighted within-year ROC-AUC (only positive/negative pairs from the same survey year) and the macro average
    over years with both classes; also returns the per-year AUCs."""
    num = den = 0.0; per = {}
    for v in np.unique(year):
        m = year == v; yy, ss = y[m], s[m]; n1 = yy.sum(); n0 = len(yy) - n1
        if n1 == 0 or n0 == 0: per[int(v)] = np.nan; continue
        a = auc(yy, ss); per[int(v)] = a; num += a * n1 * n0; den += n1 * n0
    vals = [v for v in per.values() if not np.isnan(v)]
    return (num / den if den else np.nan), (np.mean(vals) if vals else np.nan), per


def ecdf_rank(ref, x):
    srt = np.sort(ref); return (np.searchsorted(srt, x, 'left') + np.searchsorted(srt, x, 'right')) / (2 * len(ref))


def platt(oof_score, y_oof, test_score):
    """Logistic (Platt) calibration fitted on inner out-of-fold scores only. The slope is constrained to be >= 0: if the inner
    out-of-fold association is negative, the calibrated output falls back to the training prevalence (no ranking inversion)."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(oof_score).reshape(-1, 1), y_oof)
    if lr.coef_[0, 0] < 0: return np.full(len(np.asarray(test_score)), float(np.mean(y_oof)))
    return lr.predict_proba(np.asarray(test_score).reshape(-1, 1))[:, 1]


def youden_threshold(y_oof, p_oof):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_oof, p_oof); return float(thr[np.argmax(tpr - fpr)])


def threshold_metrics(y, p, thr):
    y = np.asarray(y).astype(int); yhat = (np.asarray(p) >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum()); fp = int(((yhat == 1) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12); den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {'threshold': thr, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'sensitivity': sens, 'specificity': spec, 'balanced_accuracy': (sens + spec) / 2,
            'precision': prec, 'F1': f1, 'MCC': (tp * tn - fp * fn) / den if den > 0 else 0.0}


def calibration(y, p):
    """Brier score, calibration intercept (calibration-in-the-large, slope fixed at 1) and slope (logistic recalibration)."""
    from scipy.optimize import minimize
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p)); y = np.asarray(y, float)
    if np.ptp(lp) < 1e-12:                                                          # constant prediction (calibration fallback)
        return {'Brier': float(np.mean((p - y) ** 2)), 'calibration_intercept': float(np.log(y.mean() / (1 - y.mean())) - lp[0]), 'calibration_slope': np.nan}
    def nll(theta, offset_only):
        eta = (theta[0] + lp) if offset_only else (theta[0] + theta[1] * lp)
        return np.sum(np.logaddexp(0, eta) - y * eta)
    slope = minimize(nll, [0.0, 1.0], args=(False,), method='BFGS').x[1]          # logistic recalibration slope
    intercept = minimize(nll, [0.0], args=(True,), method='BFGS').x[0]            # calibration-in-the-large (slope fixed at 1)
    return {'Brier': float(np.mean((p - y) ** 2)), 'calibration_intercept': float(intercept), 'calibration_slope': float(slope)}


# ----------------------------------------------------------------------------------------------- inference
def corrected_t(diffs, n_train, n_test, n_folds_total):
    """Nadeau & Bengio (2003) corrected resampled t: mean, 95% CI and two-sided p for paired fold differences."""
    d = np.asarray(diffs, float); k = len(d); m = d.mean(); v = d.var(ddof=1)
    se = np.sqrt((1 / k + n_test / n_train) * v) if v > 0 else 0.0
    t = m / se if se > 0 else np.inf; q = stats.t.ppf(0.975, k - 1)
    return m, m - q * se, m + q * se, 2 * stats.t.sf(abs(t), k - 1)


def cluster_bootstrap(groups, stat_fn, B=1000, seed=20261122):
    """Location-cluster bootstrap: resample locations with replacement; stat_fn(index_array) -> value."""
    rng = np.random.default_rng(seed); ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(B):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug), replace=True)]); vals.append(stat_fn(s))
    vals = np.array(vals, float); return np.nanquantile(vals, 0.025), np.nanquantile(vals, 0.975), vals


def holm(pvals):
    p = np.asarray(pvals, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(p) - i) * p[j]); adj[j] = min(run, 1.0)
    return adj
