"""Standard benchmark functions (minimisation, optimum 0) for the optimiser correctness check.
Positions in [-1, 1]^D are mapped to each function's usual domain. 'Shifted' versions move the optimum away from the
centre of the domain to detect centre (origin) bias."""
import numpy as np

DOMAINS = {'Sphere': 100.0, 'Rastrigin': 5.12, 'Rosenbrock': 30.0, 'Ackley': 32.0}


def sphere(Z): return np.sum(Z ** 2, axis=1)
def rastrigin(Z): return 10 * Z.shape[1] + np.sum(Z ** 2 - 10 * np.cos(2 * np.pi * Z), axis=1)
def rosenbrock(Z): return np.sum(100 * (Z[:, 1:] - Z[:, :-1] ** 2) ** 2 + (1 - Z[:, :-1]) ** 2, axis=1)
def ackley(Z):
    D = Z.shape[1]
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(Z ** 2, axis=1) / D)) - np.exp(np.sum(np.cos(2 * np.pi * Z), axis=1) / D) + 20 + np.e

FUNCS = {'Sphere': sphere, 'Rastrigin': rastrigin, 'Rosenbrock': rosenbrock, 'Ackley': ackley}


def make_problem(name, D, shifted, seed=0):
    """Return a MAXIMISATION fitness on [-1, 1]^D (negative error) and the shift used."""
    R = DOMAINS[name]; f = FUNCS[name]
    o = np.random.default_rng(seed + 31 * D + len(name)).uniform(-0.8 * R, 0.8 * R, D) if shifted else np.zeros(D)
    one = 1.0 if name == 'Rosenbrock' else 0.0                    # Rosenbrock optimum at z = 1 (shifted: x = o)
    def fitness(X):
        x = X * R
        return -f(x - o + one) if shifted or name == 'Rosenbrock' else -f(x)
    return fitness, o
