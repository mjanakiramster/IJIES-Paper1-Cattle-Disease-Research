"""Revision-3 manuscript tables (Phase 7): new versions only; no model is refitted.
Table2_feature_sets_v2: FS1-FS5 nesting used in the ablation (v1 listed FS1 and FS5 only).
Table3_models_v3: corrected model table, read from R2T10 (hyperparameters taken from the instantiated objects).
   Fixes against v2: CatBoost-village is a location-level regressor with depth 4 and is used with EL and ELK;
   SVM-RBF scores are decision-function values that are ranked inside the ensemble (no Platt step inside the library);
   group-balanced CatBoost weights disclosed; LightGBM min_child_samples and lambda added; optimisation row removed (Supplement S2).
R2T11_calibration_fold_and_pooled_v1: fold-level medians (IQR) and pooled estimates [location-cluster bootstrap 95% CI]
   of the calibration slope and intercept, the Brier score and the Brier skill score against the constant prevalence forecast.
Run from the project root: python src/paper1/rev3_tables_v1.py"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; P1 = ROOT / 'results/paper1_v1'; R2 = ROOT / 'results/paper1_rev2_v1'; TAB = R2 / 'tables'
DIS = ['CBPP', 'FMD', 'LSD']; MODELS = ['FS1', 'FS5', '(a) Rank average', '(b) Inner-selected single']


def table2():
    rows = [['FS1', 'Environmental', 'Altitude, temperature, precipitation, relative humidity, wind speed', '5', '5', 'No'],
            ['FS2', '+ Climate indices', 'Sea-level temperature, De Martonne index, temperature x RH, log precipitation x RH, log precipitation, wind x RH', '6', '11', 'No'],
            ['FS3', '+ Coordinates', 'Latitude, longitude, latitude + longitude, latitude - longitude', '4', '15', 'No'],
            ['FS4', '+ Neighbourhood and cross-disease rates', 'Gaussian-kernel rates at 50, 150 and 400 km (9) and rates over the 5, 15 and 40 nearest training locations (9), for CBPP, FMD and LSD', '18', '33', 'Yes (training rows only; own 81 km block excluded)'],
            ['FS5', '+ Sampling density', 'Log kernel density of training households at 50, 150 and 400 km (3); log distance to the 5th, 15th and 40th nearest training location (3)', '6', '39', 'No (household counts and distances only)']]
    T = pd.DataFrame(rows, columns=['Feature set', 'Added group', 'Variables', 'Added', 'Total', 'Uses outcome labels?']); T.to_csv(TAB / 'Table2_feature_sets_v2.csv', index=False); return T


def table3():
    H = pd.read_csv(TAB / 'R2T10_hyperparameters_from_code_v1.csv').set_index('Model')
    def g(m, k): v = H.loc[m, k]; return int(v) if isinstance(v, float) and float(v).is_integer() else v
    et, rf, xg, cb, lg, sv = 'ExtraTrees [EL]', 'RF [B]', 'XGB [B]', 'CatBoost [B]', 'LightGBM [ELK]', 'SVM-RBF [ELK]'
    assert g('ExtraTrees (FS1-FS5 feature-set and ablation experiments)', 'n_estimators') == 300 and g('CatBoost-village [EL]', 'depth') == 4 and g('CatBoost-village [ELK]', 'depth') == 4
    rows = [['ExtraTrees (FS1-FS5)', 'Feature-set comparison and ablation', f"{g('ExtraTrees (FS1-FS5 feature-set and ablation experiments)', 'n_estimators')} trees, min_samples_leaf {g('ExtraTrees (FS1-FS5 feature-set and ablation experiments)', 'min_samples_leaf')}, max_features sqrt"],
            ['GAM (B, E, EL)', 'Library (3)', f"Cubic splines ({g('GAM [B]', 'n_knots')} knots) + L2 logistic regression, C = {g('GAM [B]', 'C')}; standardised inputs"],
            ['Random forest (B, EL)', 'Library (2)', f"{g(rf, 'n_estimators')} trees, min_samples_leaf {g(rf, 'min_samples_leaf')}, max_features sqrt"],
            ['XGBoost (B, EL)', 'Library (2)', f"{g(xg, 'n_estimators')} trees, depth {g(xg, 'max_depth')}, learning rate {g(xg, 'learning_rate')}, subsample {g(xg, 'subsample')}, colsample {g(xg, 'colsample_bytree')}, min_child_weight {g(xg, 'min_child_weight'):g}"],
            ['CatBoost (B, E, EL)', 'Library (3)', f"{g(cb, 'iterations')} iterations, depth {g(cb, 'depth')}, learning rate {g(cb, 'learning_rate')}"],
            ['CatBoost, group-balanced (EL)', 'Library (1)', 'As CatBoost; training rows weighted by 1 / size of the household\'s survey year-region group (a weight, not a predictor)'],
            ['CatBoost, village-level (EL, ELK)', 'Library (2)', f"Location-level regressor of the mean outcome, weighted by households; {g('CatBoost-village [EL]', 'iterations')} iterations, depth {g('CatBoost-village [EL]', 'depth')}, learning rate {g('CatBoost-village [EL]', 'learning_rate')}"],
            ['CatBoost, multi-task (EL)', 'Library (1)', 'Joint model of the three diseases (MultiLogloss); 300 iterations, depth 5, learning rate 0.05'],
            ['k-nearest neighbours (E)', 'Library (1)', f"k = {g('kNN [E]', 'n_neighbors')}; standardised inputs"],
            ['Logistic regression (EL, ELK)', 'Library (2)', f"L2, C = {g('LR [EL]', 'C')}; standardised inputs"],
            ['ExtraTrees (EL, ELK)', 'Library (2)', f"{g(et, 'n_estimators')} trees, min_samples_leaf {g(et, 'min_samples_leaf')}, max_features sqrt"],
            ['MLP (EL)', 'Library (1)', f"Hidden layers (32, 16), alpha {g('MLP [EL]', 'alpha')}, {g('MLP [EL]', 'max_iter')} iterations; standardised inputs"],
            ['GP kriging (G)', 'Library (1)', 'Location-level Gaussian process on empirical logits; Matern-3/2 kernel with automatic relevance determination + white noise'],
            ['LightGBM (ELK)', 'Library (1)', f"{g(lg, 'n_estimators')} trees, {g(lg, 'num_leaves')} leaves, learning rate {g(lg, 'learning_rate')}, subsample {g(lg, 'subsample')}, colsample {g(lg, 'colsample_bytree')}, min_child_samples {g(lg, 'min_child_samples')}, lambda {g(lg, 'reg_lambda'):g}"],
            ['SVM-RBF (ELK)', 'Library (1)', f"C = {g(sv, 'C'):g}, gamma = scale; standardised inputs; decision-function scores, ranked like all library outputs"],
            ['(a) Rank average', 'Ensemble', 'Equal-weight mean of the 23 ECDF ranks; Platt calibration and Youden threshold on inner out-of-fold scores'],
            ['(b) Inner-selected single', 'Comparator', 'Library model with the best mean inner-fold ROC-AUC in each outer split; Platt calibration and Youden threshold as for (a)']]
    n_lib = sum(int(r[1].split('(')[1].rstrip(')')) for r in rows if r[1].startswith('Library')); assert n_lib == 23, n_lib
    T = pd.DataFrame(rows, columns=['Model (library feature sets)', 'Role (number of library members)', 'Fixed hyperparameters']); T.to_csv(TAB / 'Table3_models_v3.csv', index=False); return T


def calibration_table():
    F = pd.read_csv(P1 / 'metrics' / 'fold_metrics_v2.csv'); M = pd.read_csv(TAB / 'R2T6_metrics_with_CI_v1.csv'); rows = []
    def parse(s): a = s.replace('[', ' ').replace(']', ' ').replace(',', ' ').split(); return float(a[0]), float(a[1]), float(a[2])
    for d in DIS:
        for m in MODELS:
            q = F[(F.disease == d) & (F.model == m)]; r = M[(M.Disease == d) & (M.Model == m)].iloc[0]; p = float(r.Prevalence); ref = p * (1 - p)
            b, blo, bhi = parse(r.Brier); s, slo, shi = parse(r['Calibration slope']); i_, ilo, ihi = parse(r['Calibration intercept'])
            qs = q.calibration_slope.quantile([0.25, 0.5, 0.75]).values; qi = q.calibration_intercept.median()
            rows.append({'Disease': d, 'Model': m, 'Calibration slope: fold median (IQR)': f'{qs[1]:.2f} ({qs[0]:.2f} to {qs[2]:.2f})', 'Calibration slope: pooled [95% CI]': f'{s:.2f} [{slo:.2f}, {shi:.2f}]',
                         'Calibration intercept: fold median': f'{qi:+.2f}', 'Calibration intercept: pooled [95% CI]': f'{i_:+.2f} [{ilo:+.2f}, {ihi:+.2f}]',
                         'Brier: pooled [95% CI]': f'{b:.3f} [{blo:.3f}, {bhi:.3f}]', 'Reference Brier p(1-p)': f'{ref:.3f}', 'Brier skill score [95% CI]': f'{1 - b / ref:+.3f} [{1 - bhi / ref:+.3f}, {1 - blo / ref:+.3f}]',
                         'Platt fallbacks (folds)': int(q.platt_fallback.sum())})
    T = pd.DataFrame(rows); T.to_csv(TAB / 'R2T11_calibration_fold_and_pooled_v1.csv', index=False); return T


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
    print(table2().to_string()); print(table3().to_string()); print(calibration_table().to_string())
