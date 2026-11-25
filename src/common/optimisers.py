"""Optimisers with one interface:  optimise(fitness, X0, max_evals, rng) -> (best_x, best_f, history)
All MAXIMISE. Positions live in [-1, 1]^D. `fitness` maps an (n, D) array to n values.
Every function evaluation, including the initial population, counts against max_evals.

SGO   — Satapathy & Naik (2016), improving + acquiring phases, c = 0.2, greedy acceptance.
MSGO  — Naik, Satapathy & Abraham (2020): acquiring phase with self-awareness probability SAP = 0.7
        (equations as in Meesala, Parida & Naik 2024, Algorithm 2).
PSO   — inertia weight 0.9 -> 0.4, c1 = c2 = 2.0, |v| <= 0.2 x range (Shi & Eberhart 1998).
GA    — real-coded: binary tournament, SBX crossover (pc 0.9, eta 15), polynomial mutation (pm 1/D, eta 20), 1 elite.
DE    — DE/rand/1/bin, F = 0.5, CR = 0.9 (Storn & Price 1997).
GWO   — Mirjalili, Mirjalili & Lewis (2014), a: 2 -> 0.
HGWOPSO — run-3 hybrid (PSO velocity pulled towards the three GWO leader-guided positions); reference only.
RANDOM — uniform random search (null baseline)."""
import numpy as np

LB, UB = -1.0, 1.0


class Budget:
    def __init__(self, f, max_evals):
        self.f, self.max, self.n = f, int(max_evals), 0
        self.best_f, self.best_x, self.history = -np.inf, None, []

    @property
    def left(self): return self.max - self.n

    def __call__(self, X):
        X = np.atleast_2d(X); out = np.full(len(X), -np.inf); k = min(len(X), self.left)
        if k > 0:
            out[:k] = self.f(X[:k]); self.n += k
            for v, x in zip(out[:k], X[:k]):
                if v > self.best_f: self.best_f, self.best_x = float(v), x.copy()
                self.history.append(self.best_f)
        return out


def clip(X): return np.clip(X, LB, UB)


# ---------- SGO / MSGO update rules (separate functions so they can be unit-tested) ----------
def sgo_improve(X, gbest, c, R):
    """Improving phase: X_new = c * X + R * (gbest - X)."""
    return c * X + R * (gbest[None, :] - X)


def sgo_acquire(X, fx, gbest, partner, R1, R2):
    """Acquiring phase (maximisation): if i is better than its partner p: X_i + R1 (X_i - X_p) + R2 (gbest - X_i);
    otherwise X_i + R1 (X_p - X_i) + R2 (gbest - X_i)."""
    Xp = X[partner]; better = (fx > fx[partner])[:, None]
    return np.where(better, X + R1 * (X - Xp), X + R1 * (Xp - X)) + R2 * (gbest[None, :] - X)


def msgo_acquire(X, fx, gbest, partner, R1, R2, u, Xrand, sap=0.7):
    """MSGO acquiring phase: as SGO, except a person better than its partner is re-drawn uniformly (Xrand) when u <= SAP."""
    new = sgo_acquire(X, fx, gbest, partner, R1, R2)
    redraw = ((fx > fx[partner]) & (u <= sap))[:, None]
    return np.where(redraw, Xrand, new)


def _partners(n, rng):
    p = rng.integers(0, n - 1, n); p[p >= np.arange(n)] += 1; return p     # random partner != self


def sgo(fitness, X0, max_evals, rng, c=0.2, modified=False, sap=0.7):
    B = Budget(fitness, max_evals); X = X0.copy(); fx = B(X); n, D = X.shape
    while B.left > 0:
        g = X[np.argmax(fx)].copy()
        Xn = clip(sgo_improve(X, g, c, rng.random((n, D)))); fn = B(Xn); acc = fn > fx; X[acc], fx[acc] = Xn[acc], fn[acc]
        if B.left <= 0: break
        g = X[np.argmax(fx)].copy(); p = _partners(n, rng); R1, R2 = rng.random((2, n, D))
        if modified: Xn = msgo_acquire(X, fx, g, p, R1, R2, rng.random(n), rng.uniform(LB, UB, (n, D)), sap)
        else: Xn = sgo_acquire(X, fx, g, p, R1, R2)
        Xn = clip(Xn); fn = B(Xn); acc = fn > fx; X[acc], fx[acc] = Xn[acc], fn[acc]
    return B.best_x, B.best_f, B.history


def msgo(fitness, X0, max_evals, rng): return sgo(fitness, X0, max_evals, rng, modified=True)


def pso(fitness, X0, max_evals, rng, w0=0.9, w1=0.4, c1=2.0, c2=2.0):
    B = Budget(fitness, max_evals); X = X0.copy(); n, D = X.shape; vmax = 0.2 * (UB - LB)
    V = np.zeros_like(X); fx = B(X); P, fp = X.copy(), fx.copy(); T = max(1, (max_evals - n) // n); t = 0
    while B.left > 0:
        w = w0 - (w0 - w1) * min(1.0, t / T); g = P[np.argmax(fp)]
        R1, R2 = rng.random((2, n, D)); V = np.clip(w * V + c1 * R1 * (P - X) + c2 * R2 * (g[None, :] - X), -vmax, vmax)
        X = clip(X + V); fx = B(X); m = fx > fp; P[m], fp[m] = X[m], fx[m]; t += 1
    return B.best_x, B.best_f, B.history


def ga(fitness, X0, max_evals, rng, pc=0.9, eta_c=15.0, eta_m=20.0):
    B = Budget(fitness, max_evals); X = X0.copy(); n, D = X.shape; fx = B(X); pm = 1.0 / D
    while B.left > 0:
        def tour():
            a, b = rng.integers(0, n, 2); return X[a] if fx[a] >= fx[b] else X[b]
        kids = []
        while len(kids) < n - 1:
            p1, p2 = tour().copy(), tour().copy()
            if rng.random() < pc:                                   # SBX
                u = rng.random(D); beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
                c1_, c2_ = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2), 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
            else: c1_, c2_ = p1, p2
            for ch in (c1_, c2_):                                   # polynomial mutation
                m = rng.random(D) < pm; u = rng.random(D)
                d = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
                kids.append(clip(np.where(m, ch + d * (UB - LB), ch)))
        kids = np.array(kids[:n - 1]); fk = B(kids); e = np.argmax(fx)
        X = np.vstack([X[e], kids]); fx = np.concatenate([[fx[e]], fk])
    return B.best_x, B.best_f, B.history


def de(fitness, X0, max_evals, rng, F=0.5, CR=0.9):
    B = Budget(fitness, max_evals); X = X0.copy(); n, D = X.shape; fx = B(X)
    while B.left > 0:
        idx = np.array([rng.choice(np.delete(np.arange(n), i), 3, replace=False) for i in range(n)])
        Vm = clip(X[idx[:, 0]] + F * (X[idx[:, 1]] - X[idx[:, 2]]))
        cross = rng.random((n, D)) < CR; cross[np.arange(n), rng.integers(0, D, n)] = True
        U = np.where(cross, Vm, X); fu = B(U); m = fu > fx; X[m], fx[m] = U[m], fu[m]
    return B.best_x, B.best_f, B.history


def gwo(fitness, X0, max_evals, rng, hybrid=False):
    B = Budget(fitness, max_evals); X = X0.copy(); n, D = X.shape; fx = B(X); V = np.zeros_like(X)
    o = np.argsort(-fx)[:3]; L, Lf = X[o].copy(), fx[o].copy(); T = max(1, (max_evals - n) // n); t = 0
    while B.left > 0:
        a = 2 - 2 * min(1.0, t / T); Xn = []
        for j in range(3):
            r1, r2 = rng.random((2, n, D)); A = 2 * a * r1 - a; Cc = 2 * r2; Xn.append(L[j] - A * np.abs(Cc * L[j] - X))
        if hybrid:
            w = 0.5 + rng.random() / 2; r = rng.random((3, n, D))
            V = np.clip(w * (V + 0.5 * r[0] * (Xn[0] - X) + 0.5 * r[1] * (Xn[1] - X) + 0.5 * r[2] * (Xn[2] - X)), -0.5, 0.5); X = clip(X + V)
        else:
            X = clip((Xn[0] + Xn[1] + Xn[2]) / 3)
        fx = B(X); allX = np.vstack([L, X]); allf = np.concatenate([Lf, fx]); o = np.argsort(-allf)[:3]; L, Lf = allX[o].copy(), allf[o].copy(); t += 1
    return B.best_x, B.best_f, B.history


def hgwopso(fitness, X0, max_evals, rng): return gwo(fitness, X0, max_evals, rng, hybrid=True)


def random_search(fitness, X0, max_evals, rng):
    B = Budget(fitness, max_evals); B(X0)
    while B.left > 0: B(rng.uniform(LB, UB, (min(len(X0), B.left), X0.shape[1])))
    return B.best_x, B.best_f, B.history


OPTIMISERS = {'SGO': sgo, 'MSGO': msgo, 'PSO': pso, 'GA': ga, 'DE': de, 'GWO': gwo, 'HGWOPSO': hgwopso, 'RANDOM': random_search}
