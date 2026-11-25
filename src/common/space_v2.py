"""Reduced SR-SGO search space (Stage 6 design revision): 15 variables in [-1, 1]. Climate variables and the coordinate
group are always on. Binary variables: bit = 1 if x > 0 (S-shaped transfer, threshold 0.5)."""
import numpy as np
from .features import CLIMATE_NAMES

SWITCHES = ['kernel_rates', 'knn_rates', 'idw_rates', 'density_terms', 'cross_disease_rates']
CONT = ['bandwidth_1_km', 'bandwidth_2_km', 'bandwidth_3_km', 'k_1', 'k_2', 'k_3', 'prior_strength', 'idw_power', 'min_samples_leaf', 'max_features']
NAMES = SWITCHES + CONT; DIM = len(NAMES)                      # 15
_LOG = {'bw': (25.0, 400.0), 'prior': (0.5, 20.0)}


def _u(x): return (np.clip(x, -1, 1) + 1) / 2
def _x(u): return 2 * np.clip(u, 0, 1) - 1
def _logmap(u, lo, hi): return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
def _loginv(v, lo, hi): return (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo))


def decode(x):
    x = np.asarray(x, float); bits = x[:5] > 0; u = _u(x[5:])
    spec = {'climate': list(CLIMATE_NAMES), 'coordinates': True, **{s: bool(b) for s, b in zip(SWITCHES, bits)},
            'bandwidths': tuple(sorted({round(_logmap(u[i], *_LOG['bw']), 1) for i in range(3)})),
            'ks': tuple(sorted({int(round(3 + u[3 + i] * 57)) for i in range(3)})), 'prior': _logmap(u[6], *_LOG['prior']),
            'idw_power': 1 + 2 * float(u[7]), 'min_samples_leaf': int(round(2 + u[8] * 48)), 'max_features': 0.1 + 0.9 * float(u[9])}
    spec['k'] = spec['ks'][len(spec['ks']) // 2]; spec['complexity'] = float(bits.sum()) / 5; spec['repaired'] = False
    return spec


def anchor_fs5(n_features=39):
    """Position encoding the fixed FS5 configuration (Stage 3)."""
    bits = [0.5, 0.5, -0.5, 0.5, 0.5]                                   # kernel, knn on; idw off; density, cross-disease on
    u = [_loginv(50, *_LOG['bw']), _loginv(150, *_LOG['bw']), _loginv(400, *_LOG['bw']), (5 - 3) / 57, (15 - 3) / 57, (40 - 3) / 57,
         _loginv(1.0, *_LOG['prior']), 0.5, (10 - 2) / 48, (1 / np.sqrt(n_features) - 0.1) / 0.9]
    return np.r_[bits, _x(np.array(u))]
