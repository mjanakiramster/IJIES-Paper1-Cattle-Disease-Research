"""Pairwise relay (alternating-generation) hybrids of the six base optimisers — exploratory addition, 2026-11-21.

Framework (identical for all 15 pairs, so the pairs are comparable):
  * one shared population X (n x D) with fitness fx, plus shared memories used by the components that need them:
    personal bests P/fp (PSO), velocities V (PSO), the three best solutions ever seen L/Lf (GWO leaders);
  * the global best g is the best solution evaluated so far (Budget.best_x);
  * generations alternate: even generations apply component A's own update and replacement rule, odd generations B's;
  * every evaluation, including the initial population, counts against the shared budget (common.optimisers.Budget);
  * time-varying parameters (PSO inertia 0.9 -> 0.4, GWO a 2 -> 0) follow the fraction of the budget used.
Component steps reuse the update equations of common.optimisers (SGO c = 0.2, MSGO SAP = 0.7, PSO c1 = c2 = 2 with
|v| <= 0.2 x range, GA binary tournament + SBX(0.9, 15) + polynomial mutation (1/D, 20) + 1 elite, DE/rand/1/bin
F = 0.5 CR = 0.9, GWO three-leader update). All MAXIMISE on [-1, 1]^D."""
from itertools import combinations
import numpy as np
from .optimisers import Budget, clip, sgo_improve, sgo_acquire, msgo_acquire, _partners, LB, UB

BASE = ['SGO', 'MSGO', 'PSO', 'GA', 'DE', 'GWO']
PAIRS = [f'{a}-{b}' for a, b in combinations(BASE, 2)]          # 15 hybrids


def _greedy(S, Xn, fn):
    acc = fn > S['fx']; S['X'][acc], S['fx'][acc] = Xn[acc], fn[acc]


def _sgo_step(S, B, rng, modified=False, c=0.2, sap=0.7):
    n, D = S['X'].shape
    Xn = clip(sgo_improve(S['X'], B.best_x, c, rng.random((n, D)))); _greedy(S, Xn, B(Xn))
    if B.left <= 0: return
    p = _partners(n, rng); R1, R2 = rng.random((2, n, D))
    Xn = msgo_acquire(S['X'], S['fx'], B.best_x, p, R1, R2, rng.random(n), rng.uniform(LB, UB, (n, D)), sap) if modified \
        else sgo_acquire(S['X'], S['fx'], B.best_x, p, R1, R2)
    Xn = clip(Xn); _greedy(S, Xn, B(Xn))


def _pso_step(S, B, rng, w0=0.9, w1=0.4, c1=2.0, c2=2.0):
    n, D = S['X'].shape; w = w0 - (w0 - w1) * min(1.0, B.n / B.max); vmax = 0.2 * (UB - LB)
    R1, R2 = rng.random((2, n, D))
    S['V'] = np.clip(w * S['V'] + c1 * R1 * (S['P'] - S['X']) + c2 * R2 * (B.best_x[None, :] - S['X']), -vmax, vmax)
    S['X'] = clip(S['X'] + S['V']); S['fx'] = B(S['X'])


def _ga_step(S, B, rng, pc=0.9, eta_c=15.0, eta_m=20.0):
    X, fx = S['X'], S['fx']; n, D = X.shape; pm = 1.0 / D
    def tour():
        a, b = rng.integers(0, n, 2); return X[a] if fx[a] >= fx[b] else X[b]
    kids = []
    while len(kids) < n - 1:
        p1, p2 = tour().copy(), tour().copy()
        if rng.random() < pc:
            u = rng.random(D); beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
            c1_, c2_ = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2), 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        else: c1_, c2_ = p1, p2
        for ch in (c1_, c2_):
            m = rng.random(D) < pm; u = rng.random(D)
            d = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
            kids.append(clip(np.where(m, ch + d * (UB - LB), ch)))
    kids = np.array(kids[:n - 1]); fk = B(kids); e = int(np.argmax(fx))
    S['X'] = np.vstack([X[e], kids]); S['fx'] = np.concatenate([[fx[e]], fk])


def _de_step(S, B, rng, F=0.5, CR=0.9):
    X = S['X']; n, D = X.shape
    idx = np.array([rng.choice(np.delete(np.arange(n), i), 3, replace=False) for i in range(n)])
    Vm = clip(X[idx[:, 0]] + F * (X[idx[:, 1]] - X[idx[:, 2]]))
    cross = rng.random((n, D)) < CR; cross[np.arange(n), rng.integers(0, D, n)] = True
    U = np.where(cross, Vm, X); _greedy(S, U, B(U))


def _gwo_step(S, B, rng):
    X = S['X']; n, D = X.shape; a = 2 - 2 * min(1.0, B.n / B.max); Xn = []
    for j in range(3):
        r1, r2 = rng.random((2, n, D)); A = 2 * a * r1 - a; Cc = 2 * r2; Xn.append(S['L'][j] - A * np.abs(Cc * S['L'][j] - X))
    S['X'] = clip((Xn[0] + Xn[1] + Xn[2]) / 3); S['fx'] = B(S['X'])


STEPS = {'SGO': _sgo_step, 'MSGO': lambda S, B, rng: _sgo_step(S, B, rng, modified=True), 'PSO': _pso_step,
         'GA': _ga_step, 'DE': _de_step, 'GWO': _gwo_step}


def _update_memories(S):
    m = S['fx'] > S['fp']; S['P'][m], S['fp'][m] = S['X'][m], S['fx'][m]
    allX = np.vstack([S['L'], S['X']]); allf = np.concatenate([S['Lf'], S['fx']]); o = np.argsort(-allf)[:3]
    S['L'], S['Lf'] = allX[o].copy(), allf[o].copy()


def relay_hybrid(a, b, fitness, X0, max_evals, rng, trace=None):
    """Alternate component a (even generations) and b (odd generations) on one shared population."""
    B = Budget(fitness, max_evals); X = X0.copy(); fx = B(X)
    o = np.argsort(-fx)[:3]
    S = {'X': X, 'fx': fx, 'P': X.copy(), 'fp': fx.copy(), 'V': np.zeros_like(X), 'L': X[o].copy(), 'Lf': fx[o].copy()}
    t = 0
    while B.left > 0:
        comp = a if t % 2 == 0 else b
        if trace is not None: trace.append(comp)
        STEPS[comp](S, B, rng); _update_memories(S); t += 1
    return B.best_x, B.best_f, B.history


def make_hybrid(name):
    a, b = name.split('-')
    def opt(fitness, X0, max_evals, rng): return relay_hybrid(a, b, fitness, X0, max_evals, rng)
    opt.__name__ = f'hybrid_{a}_{b}'; return opt


HYBRIDS = {p: make_hybrid(p) for p in PAIRS}
