"""SR-SGO decision vector (24 variables, each position in [-1, 1]) and its decoding into a feature/learner specification.
Binary variables use the pre-registered S-shaped transfer with threshold 0.5, i.e. bit = 1 if sigmoid(x) > 0.5 (x > 0)."""
import numpy as np
from .features import CLIMATE_NAMES

SWITCHES = ['kernel_rates', 'knn_rates', 'idw_rates', 'density_terms', 'cross_disease_rates']
N_CLIM = len(CLIMATE_NAMES)                       # 11
BITS = N_CLIM + 1 + len(SWITCHES)                 # 17 switchable groups (complexity denominator)
CONT = ['bandwidth_1_km', 'bandwidth_2_km', 'k_nearest_locations', 'prior_strength', 'idw_power', 'min_samples_leaf', 'max_features']
DIM = BITS + len(CONT)                            # 24
NAMES = CLIMATE_NAMES + ['coordinates_group'] + SWITCHES + CONT


def _unit(x):   # [-1, 1] -> [0, 1]
    return (np.clip(x, -1, 1) + 1) / 2


def _log_range(u, lo, hi):
    return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))


def decode(x):
    """Return a dict spec. Repair rule (pre-registered here): if no feature group is on, the 5 original variables are used."""
    x = np.asarray(x, float); bits = x[:BITS] > 0; u = _unit(x[BITS:])
    spec = {'climate': [n for n, b in zip(CLIMATE_NAMES, bits[:N_CLIM]) if b], 'coordinates': bool(bits[N_CLIM]),
            **{s: bool(b) for s, b in zip(SWITCHES, bits[N_CLIM + 1:])},
            'bandwidths': tuple(sorted({round(_log_range(u[0], 25, 400), 1), round(_log_range(u[1], 25, 400), 1)})),
            'k': int(round(3 + u[2] * 57)), 'prior': _log_range(u[3], 0.5, 20), 'idw_power': 1 + 2 * float(u[4]),
            'min_samples_leaf': int(round(2 + u[5] * 48)), 'max_features': 0.2 + 0.8 * float(u[6])}
    has_nbr = spec['kernel_rates'] or spec['knn_rates'] or spec['idw_rates']
    spec['repaired'] = False
    if not spec['climate'] and not spec['coordinates'] and not has_nbr:
        spec['climate'] = CLIMATE_NAMES[:5]; spec['repaired'] = True
    spec['complexity'] = float(bits.sum()) / BITS
    return spec


def random_positions(n, rng):
    return rng.uniform(-1, 1, (n, DIM))
