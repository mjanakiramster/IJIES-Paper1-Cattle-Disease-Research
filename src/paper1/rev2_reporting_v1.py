"""Revision-2 reporting helpers: (1) hyperparameter table read directly from the instantiated model objects, (2) export of
fold assignments and household-level predictions of every experiment to portable CSV files. Writes results/paper1_rev2_v1/.
Run from the project root: python src/paper1/rev2_reporting_v1.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from paper1.core import setup, extratrees, load_library, DIS
from paper1.experiments import _lgbm, _svm, _blocks
ROOT = Path(__file__).resolve().parents[2]; P1 = ROOT / 'results/paper1_v1'; REV = ROOT / 'results/paper1_rev_v1'; R2 = ROOT / 'results/paper1_rev2_v1'
KEYS = ['n_estimators', 'iterations', 'max_depth', 'depth', 'min_samples_leaf', 'max_features', 'learning_rate', 'subsample', 'colsample_bytree', 'min_child_weight', 'num_leaves',
        'min_child_samples', 'reg_lambda', 'C', 'gamma', 'kernel', 'n_neighbors', 'hidden_layer_sizes', 'alpha', 'max_iter', 'n_knots', 'degree', 'random_state', 'random_seed']


def _params(est):
    out = {}
    steps = est.steps if hasattr(est, 'steps') else [('model', est)]
    for name, step in steps:
        p = step.get_params()
        if name == 'standardscaler': out['standardised inputs'] = 'yes'; continue
        for k in KEYS:
            if k in p and p[k] is not None and not callable(p[k]): out[k] = p[k]
    return out


def hyperparameter_table():
    G = load_library(); rows = []
    rows.append({'Model': 'ExtraTrees (FS1-FS5 feature-set and ablation experiments)', 'Library feature set': '-', **_params(extratrees())})
    for name, kind, fs in G['LIB']:
        if kind == 'single': est = G['make_single'](name)
        elif kind in ('gb', 'multi'): est = G['cb']()
        elif kind == 'loc':
            from catboost import CatBoostRegressor
            est = CatBoostRegressor(iterations=300, depth=4, learning_rate=0.05, verbose=0, thread_count=1, random_seed=G['SEED'], allow_writing_files=False)
        else: est = None
        extra = {'gb': 'sample weights = inverse size of the household\'s year-region group (training rows only)', 'loc': 'location-level regressor on mean outcome, weighted by households',
                 'multi': 'multi-label (MultiLogloss) model for the three diseases', 'gp': 'location-level Gaussian process on empirical logits; Matern-3/2 ARD kernel + white noise'}.get(kind, '')
        rows.append({'Model': name, 'Library feature set': fs, **(_params(est) if est is not None else {}), 'note': extra})
    rows.append({'Model': 'LightGBM [ELK]', 'Library feature set': 'ELK', **_params(_lgbm())}); rows.append({'Model': 'SVM-RBF [ELK]', 'Library feature set': 'ELK', **_params(_svm()), 'note': 'decision_function scores (ranked within the ensemble)'})
    T = pd.DataFrame(rows).fillna(''); (R2 / 'tables').mkdir(parents=True, exist_ok=True); T.to_csv(R2 / 'tables' / 'R2T10_hyperparameters_from_code_v1.csv', index=False); return T


def export_all():
    df, Y, _, _ = setup(); EX = R2 / 'exports'; EX.mkdir(parents=True, exist_ok=True); base = df[['ID', 'Location_id', 'Year']].rename(columns={'ID': 'household_ID'})
    folds = df[['ID', 'Location_id', 'Year', 'random_fold'] + [f'{c}_r{r}' for r in (1, 2, 3) for c in ('block', 'spatial_fold')]].rename(columns={'ID': 'household_ID'})
    for bkm in (120.0, 160.0):
        bl = _blocks(df, bkm)
        for r in (1, 2, 3): folds[f'block{int(bkm)}_r{r}'] = bl[r][0]; folds[f'spatial_fold{int(bkm)}_r{r}'] = bl[r][1]
    folds['temporal_set'] = df.temporal_set.values; folds.to_csv(EX / 'fold_assignments_v1.csv', index=False)
    rows = []
    for i in range(5):
        z = np.load(REV / 'E9_random_cv' / f'random_{i}.npz'); te = z['test_idx']
        for d, dn in enumerate(DIS):
            for s, name in enumerate(('FS1', 'FS5')): rows.append(pd.DataFrame({'scheme': 'random', 'fold': i, 'disease': dn, 'model': name, 'row': te, 'observed': Y[te, d], 'score': z['fs_test'][:, d, s]}))
    for r in (1, 2, 3):
        for f in range(5):
            z = np.load(R2 / 'E11_ablation' / f'r{r}_f{f}.npz'); te = z['test_idx']
            for d, dn in enumerate(DIS):
                for s, name in enumerate(z['sets']): rows.append(pd.DataFrame({'scheme': f'spatial_r{r}', 'fold': f, 'disease': dn, 'model': str(name), 'row': te, 'observed': Y[te, d], 'score': z['test'][:, d, s]}))
            z = np.load(R2 / 'E10b_campaign_predictions' / f'r{r}_f{f}.npz'); te = z['test_idx']
            for d, dn in enumerate(DIS):
                for s, name in enumerate(('FS1', 'FS5')): rows.append(pd.DataFrame({'scheme': f'spatial_r{r}', 'fold': f, 'disease': dn, 'model': f'campaign-specific {name}', 'row': te, 'observed': Y[te, d], 'score': z['campaign_test'][:, d, s]}))
    z = np.load(P1 / 'E7_temporal' / 'forward_temporal_predictions_v1.npz'); te = np.where(df.temporal_set.values == 'test')[0]
    for d, dn in enumerate(DIS):
        for s, name in enumerate(('FS1', 'FS5')): rows.append(pd.DataFrame({'scheme': 'forward temporal', 'fold': 0, 'disease': dn, 'model': name, 'row': te, 'observed': Y[te, d], 'score': z['test'][:, d, s]}))
    Pdf = pd.concat(rows, ignore_index=True); Pdf = pd.concat([base.iloc[Pdf.row.values].reset_index(drop=True), Pdf.drop(columns='row')], axis=1)
    Pdf.to_csv(EX / 'predictions_random_ablation_campaign_temporal_v1.csv', index=False, float_format='%.6f'); return folds.shape, Pdf.shape


if __name__ == '__main__':
    print(hyperparameter_table().to_string()); print(export_all())
