"""
Run one named experiment:  python run.py <experiment> <r> [seed]

Experiments (see README section 4):
  vanilla     plain PINN, raw nondimensional units, uniform random collocation
  supervised  same network, fit directly to the exact solution (no physics)
  normalized  vanilla + each loss term scaled by its variable's size
  rad         normalized + residual-based adaptive resampling
  visc        normalized + artificial viscosity, decayed over training
  logrho      normalized + network predicts log(rho)
  causal      normalized + causal time weighting (Wang, Sankaran & Perdikaris 2024)
  fix         normalized + log(rho) + viscosity curriculum
"""
import sys
from pinn import Config, train

EXPERIMENTS = {
    "vanilla":    dict(),
    "supervised": dict(mode="supervised"),
    "normalized": dict(normalize=True),
    "rad":        dict(normalize=True, adaptive=True),
    "visc":       dict(normalize=True, nu=5e-3, nu_final=1e-3),
    "logrho":     dict(normalize=True, logrho=True),
    "causal":     dict(normalize=True, causal=50.0),
    "fix":        dict(normalize=True, logrho=True, nu=5e-3, nu_final=1e-3),
}

if __name__ == "__main__":
    name = sys.argv[1]
    r = float(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    cfg = Config(name=name, r=r, seed=seed, iters=10000, **EXPERIMENTS[name])
    train(cfg)
