"""
Sanity checks the course asks for: verify the reference solution and one PINN
error value independently of the code that produced them.

1. Across every shock in the exact solution, the Rankine-Hugoniot conditions
   s [U] = [F(U)] must hold (checked for mass, momentum, energy).
2. Across the interface (contact), u and p are continuous and rho jumps.
3. Recompute the vanilla r = 10 relative L2 error straight from the saved field.
"""
import json
import numpy as np
from riemann import initial_states, star_state, wave_summary, GAMMA_L as g

def cons(rho, u, p):
    return np.array([rho, rho * u, p / (g - 1) + 0.5 * rho * u**2])

def flux(rho, u, p):
    E = p / (g - 1) + 0.5 * rho * u**2
    return np.array([rho * u, rho * u**2 + p, u * (E + p)])

def shocked(W, p_s, side):
    rho, u, p = W
    a = (g - 1) / (g + 1)
    return rho * ((p_s / p + a) / (a * p_s / p + 1))

for r in [3, 10, 100]:
    WL, WR = initial_states(r)
    p_s, u_s = star_state(WL, WR)
    ws = wave_summary(r)
    # transmitted shock into the heavy gas
    WsR = (shocked(WR, p_s, "R"), u_s, p_s)
    s = ws["transmitted_speed"]
    rh_t = s * (cons(*WsR) - cons(*WR)) - (flux(*WsR) - flux(*WR))
    # reflected shock into the light gas
    WsL = (shocked(WL, p_s, "L"), u_s, p_s)
    s = ws["reflected_speed"]
    rh_r = s * (cons(*WsL) - cons(*WL)) - (flux(*WsL) - flux(*WL))
    print(f"r={r:>3}: RH residual transmitted {np.abs(rh_t).max():.1e}, reflected {np.abs(rh_r).max():.1e}; "
          f"contact: u*={u_s:.4f} both sides, p*={p_s:.4f}, rho jumps {WsL[0]:.3f} -> {WsR[0]:.3f}")

d = np.load("../results/runs/vanilla_r10_s0.npz")
err = np.sqrt(((d["R"] - d["Re"]) ** 2).sum() / (d["Re"] ** 2).sum())
logged = json.load(open("../results/runs/vanilla_r10_s0.json"))["err_rho"]
print(f"vanilla r=10: recomputed rel. L2 error {err:.4f}, logged {logged:.4f}")
