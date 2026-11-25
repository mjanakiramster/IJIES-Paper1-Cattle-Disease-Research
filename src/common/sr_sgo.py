"""SR-SGO optimiser (Stage 6 design revision). A new module, so the Stage 5 code stays unchanged.

Original SGO improving phase (Satapathy & Naik 2016):   X_new = c X + R (gbest - X)          -> pulls towards the origin
SR-SGO candidates (shift-invariant):
  'centroid':  X_new = m + c (X - m) + R (gbest - X),  m = population mean   (equals SGO when m = 0)
  'attract' :  X_new = X + R (gbest - X)
Acquiring phase: unchanged SGO rule (differences of positions; already translation-invariant). Greedy acceptance.
Interface and budget accounting are identical to common.optimisers."""
import numpy as np
from .optimisers import Budget, clip, sgo_acquire, _partners, LB, UB


def srsgo_improve(X, gbest, c, R, variant='centroid'):
    if variant == 'centroid':
        m = X.mean(0, keepdims=True); return m + c * (X - m) + R * (gbest[None, :] - X)
    if variant == 'attract':
        return X + R * (gbest[None, :] - X)
    raise ValueError(variant)


def sr_sgo(fitness, X0, max_evals, rng, c=0.2, variant='centroid'):
    B = Budget(fitness, max_evals); X = X0.copy(); fx = B(X); n, D = X.shape
    while B.left > 0:
        g = X[np.argmax(fx)].copy()
        Xn = clip(srsgo_improve(X, g, c, rng.random((n, D)), variant)); fn = B(Xn); acc = fn > fx; X[acc], fx[acc] = Xn[acc], fn[acc]
        if B.left <= 0: break
        g = X[np.argmax(fx)].copy(); p = _partners(n, rng); R1, R2 = rng.random((2, n, D))
        Xn = clip(sgo_acquire(X, fx, g, p, R1, R2)); fn = B(Xn); acc = fn > fx; X[acc], fx[acc] = Xn[acc], fn[acc]
    return B.best_x, B.best_f, B.history


def sr_sgo_centroid(fitness, X0, max_evals, rng): return sr_sgo(fitness, X0, max_evals, rng, variant='centroid')
def sr_sgo_attract(fitness, X0, max_evals, rng): return sr_sgo(fitness, X0, max_evals, rng, variant='attract')
