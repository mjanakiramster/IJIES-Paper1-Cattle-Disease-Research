"""Audit A2 follow-up (2026-11-21): investigate every kNN-rate mismatch between the code and the reference implementation.
Coordinates are given to 0.1 degree, so many candidate locations are at EXACTLY the same distance from a query. The code
orders candidates by its numpy distances, the reference by math.* distances (difference <= 1e-12 km), so exactly tied
neighbours at the k-th position can be ordered differently. For every mismatching (split, location, k) this script checks
(1) that the two neighbour sets differ only by locations whose reference distance equals the k-th distance (tie), and
(2) that the code's value is reproduced exactly when the reference uses the code's tie order (all else independent).
Development splits only. Run from the project root."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
SRC = Path(__file__).resolve().parent; sys.path.insert(0, str(SRC)); ROOT = SRC.parent
from common.data import load, labels, SEED
from common.features import Neighbourhood
from common.folds import inner_spatial_folds
from audit_common_v1 import Reference, load_run3
OUT = ROOT / 'results/audit_v1/A2_A3_A8'
df = load(ROOT / 'data/raw'); Y = labels(df); loc = df['loc'].values; NB = Neighbourhood(df, Y); REF = Reference(loc, df.Latitude.values, df.Longitude.values)
Dref = np.array([[REF.dist[(i, j)] for j in range(462)] for i in range(462)])
R3 = load_run3(); Dold = R3['DLOC']
splits = [(f'dev r{r} f{f}', np.where(df[f'spatial_fold_r{r}'].values != f)[0], f'block_r{r}') for r in (1, 2, 3) for f in range(5)]
splits += [(f'inner r1 f0 pair {k}', fit, 'block_r1') for k, (fit, _) in enumerate(inner_spatial_folds(splits[0][1], df.block_r1.values, SEED + 1, k=5, repeats=1))]
rows = []
for code, Dcode in (('new', NB.D), ('old run-3', Dold)):
    for name, fit, bc in (splits if code == 'new' else splits[:15]):
        bl = NB.block_of_loc(df[bc].values); n = np.bincount(loc[fit], minlength=462).astype(float)
        pos = np.column_stack([np.bincount(loc[fit], weights=Y[fit, k], minlength=462) for k in range(3)])
        if code == 'new':
            same = bl[:, None] == bl[None, :]; p0 = (pos.sum(0)[None, :] - same.astype(float) @ pos) / np.maximum(n.sum() - same.astype(float) @ n, 1.0)[:, None]
        else:
            p0 = np.repeat((pos.sum(0) / n.sum())[None, :], 462, 0)
        for x in np.unique(loc):
            cand = [j for j in range(462) if bl[j] != bl[x] and n[j] > 0]
            if not cand: continue
            oc = [cand[i] for i in np.argsort([Dcode[x, j] for j in cand], kind='stable')]; orf = sorted(cand, key=lambda j: (Dref[x, j], j))
            for k in (5, 15, 40):
                a, b = oc[:k], orf[:k]
                if set(a) == set(b): continue
                dk = Dref[x, b[-1]]; diff = set(a) ^ set(b)
                code_val = (pos[a].sum(0) + p0[x]) / (n[a].sum() + 1.0); ref_val = (pos[b].sum(0) + p0[x]) / (n[b].sum() + 1.0)
                rows.append({'code': code, 'split': name, 'location': int(x), 'k': k, 'differing_neighbours': len(diff),
                             'all_differences_are_exact_ties': bool(all(abs(Dref[x, j] - dk) < 1e-9 for j in diff)),
                             'tied_candidates_at_kth_distance': int(sum(abs(Dref[x, j] - dk) < 1e-9 for j in cand)),
                             'max_abs_value_difference': float(np.abs(code_val - ref_val).max())})
T = pd.DataFrame(rows); T.to_csv(OUT / 'A2b_knn_tie_cases_v1.csv', index=False)
S = T.groupby('code').agg(cases=('k', 'size'), explained_by_exact_ties=('all_differences_are_exact_ties', 'sum'), max_value_difference=('max_abs_value_difference', 'max')).reset_index()
S.to_csv(OUT / 'A2b_knn_tie_summary_v1.csv', index=False)
share = float(np.mean([(np.abs(Dref[i][None, :] - Dref[i][:, None]) < 1e-9).sum() > 462 for i in range(462)]))
json.dump({'coordinate_precision_degrees': 0.1, 'locations_with_at_least_one_exact_distance_tie': share,
           'max_abs_numpy_vs_math_distance_km': float(np.abs(NB.D - Dref).max()), 'summary': S.to_dict('records')}, open(OUT / 'A2b_knn_tie_summary_v1.json', 'w'), indent=1)
print(S.to_string(index=False))
