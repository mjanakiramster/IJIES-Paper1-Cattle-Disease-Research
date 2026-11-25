"""Stage 5 diagnostic — (1) within-fold correlation of inner fitness and outer ROC-AUC across all 21 runs of each
(disease, fold); (2) inner fitness of the fixed FS5 configuration under the SAME fitness used by the optimisers.
Development repeat 1 only. Run from the project root:  python src/05b_inner_outer_diagnostic.py"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.data import load, labels, ENV, SEED
from common.features import climate_coords, Neighbourhood, CLIMATE_NAMES
from common.fitness import FeatureBuilder, robust_fitness
from common.folds import inner_spatial_folds
warnings.filterwarnings('ignore')
OUT = Path('results/stage5_v1')
R = pd.read_csv(OUT / 'optimiser_real_runs.csv')
w = []
for (d, f), g in R.groupby(['disease', 'fold']):
    w.append({'disease': d, 'fold': f, 'runs': len(g), 'spearman_inner_vs_outer': stats.spearmanr(g.best_inner_fitness, g.outer_ROC_AUC).correlation})
W = pd.DataFrame(w)
df = load('data/raw'); Y = labels(df); FB = FeatureBuilder(df, Y, Neighbourhood(df, Y), climate_coords(df))
FS5 = {'climate': CLIMATE_NAMES, 'coordinates': True, 'kernel_rates': True, 'knn_rates': True, 'idw_rates': False, 'density_terms': True, 'cross_disease_rates': True,
       'bandwidths': (50.0, 150.0, 400.0), 'ks': (5, 15, 40), 'k': 15, 'prior': 1.0, 'idw_power': 2.0, 'min_samples_leaf': 10, 'max_features': 'sqrt', 'complexity': 16 / 17}
s3 = pd.read_csv('results/stage3_v1/feature_set_fold_metrics.csv'); s3 = s3[(s3.rep == 1) & (s3.feature_set == 'FS5 + cross-disease rates')]
rows = []
for di, d in enumerate(['CBPP', 'FMD', 'LSD']):
    for f in range(5):
        tr = np.where(df.spatial_fold_r1.values != f)[0]; pairs = inner_spatial_folds(tr, df.block_r1.values, SEED + 100 * f + di, k=5, repeats=1)
        F = robust_fitness(FS5, FB, pairs, df.block_r1.values, di, kind='row')
        g = R[(R.disease == d) & (R.fold == f)]
        rows.append({'disease': d, 'fold': f, 'FS5_inner_fitness': F, 'optimisers_best_inner_fitness_mean': g.best_inner_fitness.mean(),
                     'FS5_outer_ROC_AUC': float(s3[(s3.disease == d) & (s3.fold == f)].ROC_AUC.iloc[0]), 'optimisers_outer_ROC_AUC_mean': g.outer_ROC_AUC.mean()})
D = pd.DataFrame(rows).merge(W, on=['disease', 'fold']); D.to_csv(OUT / 'inner_outer_diagnostic.csv', index=False)
S = D.groupby('disease')[['FS5_inner_fitness', 'optimisers_best_inner_fitness_mean', 'FS5_outer_ROC_AUC', 'optimisers_outer_ROC_AUC_mean', 'spearman_inner_vs_outer']].mean()
S['folds_where_search_beats_FS5_on_inner'] = D.assign(b=D.optimisers_best_inner_fitness_mean > D.FS5_inner_fitness).groupby('disease').b.sum()
S['folds_where_search_beats_FS5_on_outer'] = D.assign(b=D.optimisers_outer_ROC_AUC_mean > D.FS5_outer_ROC_AUC).groupby('disease').b.sum()
S.to_csv(OUT / 'inner_outer_diagnostic_summary.csv')
pd.set_option('display.width', 250); print(S.round(4).to_string()); print('mean within-fold Spearman (all 15 folds):', round(W.spearman_inner_vs_outer.mean(), 3))
