"""P4 verification (2026-11-21): deterministic kNN tie-break vs an independent reference using the same tie rule,
for the new pipeline (FS5 + IDW) and the corrected old 23-model library code, on the 15 development splits."""
import sys, json, os
from pathlib import Path
import numpy as np
SRC = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(SRC)); ROOT = SRC.parent
from paper1.core import setup, dev_splits, load_library, TIE_DECIMALS, OUT
df, Y, NB, FB = setup(); loc = df['loc'].values
src = open(SRC / 'audit_common_v1.py').read(); key = "order = sorted(cand, key=lambda j: (self.dist[(x, j)], j))"; assert key in src
ns = {'__file__': str(SRC / 'audit_common_v1.py'), '__name__': 'reference_v2'}; exec(compile(src.replace(key, f"order = sorted(cand, key=lambda j: (round(self.dist[(x, j)], {TIE_DECIMALS}), j))"), 'reference_v2', 'exec'), ns)
R2 = ns['Reference'](loc, df.Latitude.values, df.Longitude.values)
cfg = dict(bandwidths=(50.0, 150.0, 400.0), ks=(5, 15, 40), prior=1.0, idw_power=2.0, density=True, diseases=(0, 1, 2))
res = {'tie_rule': f'(round(distance_km, {TIE_DECIMALS}), location ID)', 'splits': '15 development outer splits'}
mx = 0.0; mism = 0; nvals = 0
for r, f, tr, te, bc in dev_splits(df):
    Ff, Fp, nm = NB.rates(tr, te, df[bc].values, **cfg); ref, rnm = R2.rates(loc, Y, df[bc].values, tr, prior_mode='exclude_own_block', **cfg); assert rnm == nm
    d = np.abs(np.vstack([Ff, Fp]) - np.vstack([R2.rows(ref, loc, tr), R2.rows(ref, loc, te)])); mx = max(mx, d.max()); mism += int((d > 1e-10).sum()); nvals += d.size
res.update(new_code_values=nvals, new_code_max_abs_diff=mx, new_code_mismatches=mism)
G = load_library(); DISn = ['CBPP', 'FMD', 'LSD']; OLD = list(G['NAMES_LAG']) + list(G['NAMES_KNN']); mx2 = 0.0; mism2 = 0; n2 = 0
for r, f, tr, te, bc in dev_splits(df):
    Lf, Lp = G['lag_features'](tr, te, bc); Kf, Kp = G['knn_features'](tr, te, bc); A = np.vstack([np.column_stack([Lf, Kf]), np.column_stack([Lp, Kp])])
    ref, rnm = R2.rates(loc, Y, df[bc].values, tr, bandwidths=(50.0, 150.0, 400.0), ks=(5, 15, 40), prior=1.0, idw_power=None, density=True,
                        diseases=(0, 1, 2), prior_mode='exclude_own_block', knn_dist_ks={15}); m = {}
    for i, n in enumerate(rnm):
        if n.startswith('kern'): b, k = n[4:].split('_'); m[f'nbr_{DISn[int(k)]}_{b}km'] = i
        elif n.startswith('dens'): m[f'nbr_density_{n[4:]}km'] = i
        elif n.startswith('knndist'): m['log_dist_15th_location'] = i
        elif n.startswith('knn'): k_, k = n[3:].split('_'); m[f'knn{k_}_{DISn[int(k)]}'] = i
    idx = [m[n] for n in OLD]; ref = {l: [v[i] for i in idx] for l, v in ref.items()}
    d = np.abs(A - np.vstack([R2.rows(ref, loc, tr), R2.rows(ref, loc, te)])); mx2 = max(mx2, d.max()); mism2 += int((d > 1e-10).sum()); n2 += d.size
res.update(old_library_values=n2, old_library_max_abs_diff=mx2, old_library_mismatches=mism2)
(OUT / 'verification').mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT / 'verification' / 'P4_tiebreak_verification_v1.json', 'w'), indent=1); print(json.dumps(res, indent=1))
