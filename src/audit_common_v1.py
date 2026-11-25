"""Shared helpers for the feature-leakage audit (2026-11-21).

1. load_run3(): executes the OLD run-3 pipeline code (data/reference/run3_cattle_v5_pipeline_v3.py) up to its main loop,
   as Stage 9 did, optionally with the rows shuffled (row-order audit A8).
2. reference_rates(): an INDEPENDENT, loop-based implementation of the neighbourhood / cross-disease / density
   features written from the mathematical definitions (A2). It shares no code with common.features or the run-3 code:
   distances use math.* haversine per pair, fit statistics are accumulated with explicit Python loops, and every
   query location is processed one at a time.
   Definitions (query location x, fit locations j with >= 1 fit household and block(j) != block(x)):
     p0(x)      'exclude_own_block': sum_j pos_j / sum_j n_j over those j        (new code)
                'all_fit'          : rate over ALL fit households                 (old run-3 code)
     kernel b   (sum_j K pos_j + c p0) / (sum_j K n_j + c),  K = exp(-d^2 / (2 b^2));  density = log(1 + sum_j K n_j)
     kNN k      the k nearest such j (ties broken by location index): (sum pos + c p0) / (sum n + c);  p0 if none
                distance term = log(1 + distance to the min(k, available)-th of them), log(1 + max pair distance) if none
     IDW p      w_j = 1 / max(d, 1)^p for d > 0
"""
import math, shutil, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_run3(shuffle_seed=None, scratch='_audit_scratch'):
    src = open(ROOT / 'data/reference/run3_cattle_v5_pipeline_v3.py').read()
    src = src[:src.index("t0 = time.time()\nRES = sorted(")].replace("DATA = Path('data')", "DATA = Path('data/raw')").replace(
        "OUT = next_free_dir('results')", f"OUT = next_free_dir('{scratch}')")
    if shuffle_seed is not None:
        anchor = "df['loc'] = df.Location_id.astype(int) - 1\n"; assert anchor in src
        src = src.replace(anchor, anchor + f"df = df.sample(frac=1, random_state={int(shuffle_seed)}).reset_index(drop=True)\n")
    cwd = os.getcwd(); os.chdir(ROOT)
    try:
        R3 = {'__name__': 'run3'}; exec(compile(src, 'run3_cattle_v5_pipeline_v3.py', 'exec'), R3)
        shutil.rmtree(ROOT / R3['OUT'], ignore_errors=True)
    finally:
        os.chdir(cwd)
    return R3


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(min(max(a, 0.0), 1.0)))


class Reference:
    def __init__(self, loc, lat, lon):
        self.info = {}
        for l, a, o in zip(loc, lat, lon):                     # explicit consistency check of location coordinates
            l = int(l)
            if l in self.info: assert self.info[l] == (float(a), float(o)), f'location {l} has two coordinate pairs'
            else: self.info[l] = (float(a), float(o))
        self.locs = sorted(self.info)
        self.dist = {}
        for i in self.locs:
            for j in self.locs:
                self.dist[(i, j)] = haversine_km(*self.info[i], *self.info[j])
        self.dmax = max(self.dist.values())

    def rates(self, loc, Y, row_block, fit_idx, bandwidths, ks, prior, idw_power, density, diseases, prior_mode='exclude_own_block',
              knn_dist_ks=None, density_bandwidths=None):
        """Per-location features (dict loc -> list of values) and the column names, in the new code's column order."""
        blk = {}
        for l, b in zip(loc, row_block):
            l = int(l); b = str(b)
            if l in blk: assert blk[l] == b, f'location {l} in two blocks'
            else: blk[l] = b
        stats = {}                                              # loc -> [n, pos_0, pos_1, pos_2]
        for i in fit_idx:
            s = stats.setdefault(int(loc[i]), [0.0, 0.0, 0.0, 0.0]); s[0] += 1.0
            for k in range(3): s[1 + k] += float(Y[i, k])
        tot = [sum(s[q] for s in stats.values()) for q in range(4)]
        out, names = {}, None
        for x in self.locs:
            cand = [j for j in stats if blk[j] != blk[x] and stats[j][0] > 0]
            if prior_mode == 'exclude_own_block':
                n_c = sum(stats[j][0] for j in cand); p0 = [sum(stats[j][1 + k] for j in cand) / max(n_c, 1.0) for k in range(3)]
            else:
                p0 = [tot[1 + k] / tot[0] for k in range(3)]
            vals, nm = [], []
            for b in bandwidths:
                num = [0.0, 0.0, 0.0]; den = 0.0
                for j in cand:
                    w = math.exp(-self.dist[(x, j)] ** 2 / (2.0 * b * b)); den += w * stats[j][0]
                    for k in range(3): num[k] += w * stats[j][1 + k]
                for k in diseases: vals.append((num[k] + prior * p0[k]) / (den + prior)); nm.append(f'kern{int(b)}_{k}')
                if density: vals.append(math.log1p(den)); nm.append(f'dens{int(b)}')
            if ks:
                order = sorted(cand, key=lambda j: (self.dist[(x, j)], j))
                for k_ in ks:
                    sel = order[:int(k_)]
                    for k in diseases:
                        if sel: vals.append((sum(stats[j][1 + k] for j in sel) + prior * p0[k]) / (sum(stats[j][0] for j in sel) + prior))
                        else: vals.append(p0[k])
                        nm.append(f'knn{int(k_)}_{k}')
                    if density and (knn_dist_ks is None or k_ in knn_dist_ks):
                        vals.append(math.log1p(self.dist[(x, sel[-1])]) if sel else math.log1p(self.dmax)); nm.append(f'knndist{int(k_)}')
            if idw_power:
                num = [0.0, 0.0, 0.0]; den = 0.0
                for j in cand:
                    d = self.dist[(x, j)]
                    if d > 0:
                        w = 1.0 / max(d, 1.0) ** idw_power; den += w * stats[j][0]
                        for k in range(3): num[k] += w * stats[j][1 + k]
                for k in diseases: vals.append((num[k] + prior * p0[k]) / (den + prior)); nm.append(f'idw_{k}')
            out[x] = vals; names = nm
        return out, names

    def rows(self, per_loc, loc, idx):
        return np.array([per_loc[int(loc[i])] for i in idx], dtype=float)
