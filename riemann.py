"""
Exact solution for our minimal case: a planar shock that has just reached a
material interface (light gas on the left, heavy gas on the right).

At t = 0 the shock sits exactly on the interface, so the problem is a plain
Riemann problem: post-shock light gas (moving right) on the left, heavy gas at
rest on the right. We solve it exactly following Toro, "Riemann Solvers and
Numerical Methods for Fluid Dynamics", ch. 4. The solver allows different
gammas on each side even though our main runs use the same gamma.
"""
import numpy as np
from scipy.optimize import brentq

# ---------------------------------------------------------------- problem setup
GAMMA_L = 1.4      # light gas (air)
GAMMA_R = 1.4      # heavy gas (same gamma, so only the density jumps)
X0 = 0.4           # interface position at t = 0
T_END = 0.3        # final time; all waves stay inside [0, 1] up to this time
MS = 1.5           # incident shock Mach number


def post_shock_state(Ms=MS, g=GAMMA_L, rho1=1.0, p1=1.0):
    """Rankine-Hugoniot: state behind a right-moving shock of Mach Ms into gas at rest."""
    c1 = np.sqrt(g * p1 / rho1)
    p2 = p1 * (1 + 2 * g / (g + 1) * (Ms**2 - 1))
    rho2 = rho1 * (g + 1) * Ms**2 / ((g - 1) * Ms**2 + 2)
    Ws = Ms * c1
    u2 = Ws * (1 - rho1 / rho2)
    return rho2, u2, p2


def initial_states(r, Ms=MS):
    """Left/right states for density ratio r = rho_heavy / rho_light (pre-shock)."""
    rhoL, uL, pL = post_shock_state(Ms)
    rhoR, uR, pR = r * 1.0, 0.0, 1.0
    return (rhoL, uL, pL), (rhoR, uR, pR)


# ---------------------------------------------------------------- exact solver
def _f(p, rho, pk, g):
    """Toro's pressure function f_K and its role in the star-pressure equation."""
    A = 2 / ((g + 1) * rho)
    B = (g - 1) / (g + 1) * pk
    c = np.sqrt(g * pk / rho)
    if p > pk:   # shock
        return (p - pk) * np.sqrt(A / (p + B))
    return 2 * c / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1)   # rarefaction


def star_state(WL, WR, gL=GAMMA_L, gR=GAMMA_R):
    rhoL, uL, pL = WL
    rhoR, uR, pR = WR
    fun = lambda p: _f(p, rhoL, pL, gL) + _f(p, rhoR, pR, gR) + (uR - uL)
    p_star = brentq(fun, 1e-8, 1e4, xtol=1e-14)
    u_star = 0.5 * (uL + uR) + 0.5 * (_f(p_star, rhoR, pR, gR) - _f(p_star, rhoL, pL, gL))
    return p_star, u_star


def sample(xi, WL, WR, gL=GAMMA_L, gR=GAMMA_R, star=None):
    """Exact solution at similarity coordinate xi = (x - X0)/t. Vectorised over xi."""
    rhoL, uL, pL = WL
    rhoR, uR, pR = WR
    p_s, u_s = star if star is not None else star_state(WL, WR, gL, gR)
    cL = np.sqrt(gL * pL / rhoL)
    cR = np.sqrt(gR * pR / rhoR)
    xi = np.atleast_1d(xi).astype(float)
    rho = np.empty_like(xi); u = np.empty_like(xi); p = np.empty_like(xi)

    # ---- left wave
    if p_s > pL:   # left shock
        rho_sL = rhoL * ((p_s / pL + (gL - 1) / (gL + 1)) / ((gL - 1) / (gL + 1) * p_s / pL + 1))
        SL = uL - cL * np.sqrt((gL + 1) / (2 * gL) * p_s / pL + (gL - 1) / (2 * gL))
        left_fan = None
    else:          # left rarefaction
        rho_sL = rhoL * (p_s / pL) ** (1 / gL)
        c_sL = cL * (p_s / pL) ** ((gL - 1) / (2 * gL))
        SL = None
        left_fan = (uL - cL, u_s - c_sL)

    # ---- right wave
    if p_s > pR:   # right shock
        rho_sR = rhoR * ((p_s / pR + (gR - 1) / (gR + 1)) / ((gR - 1) / (gR + 1) * p_s / pR + 1))
        SR = uR + cR * np.sqrt((gR + 1) / (2 * gR) * p_s / pR + (gR - 1) / (2 * gR))
        right_fan = None
    else:
        rho_sR = rhoR * (p_s / pR) ** (1 / gR)
        c_sR = cR * (p_s / pR) ** ((gR - 1) / (2 * gR))
        SR = None
        right_fan = (u_s + c_sR, uR + cR)

    for i, s in enumerate(xi):
        if s <= u_s:  # left of contact
            if left_fan is None:
                if s <= SL:
                    rho[i], u[i], p[i] = rhoL, uL, pL
                else:
                    rho[i], u[i], p[i] = rho_sL, u_s, p_s
            else:
                if s <= left_fan[0]:
                    rho[i], u[i], p[i] = rhoL, uL, pL
                elif s >= left_fan[1]:
                    rho[i], u[i], p[i] = rho_sL, u_s, p_s
                else:
                    u[i] = 2 / (gL + 1) * (cL + (gL - 1) / 2 * uL + s)
                    c = 2 / (gL + 1) * (cL + (gL - 1) / 2 * (uL - s))
                    rho[i] = rhoL * (c / cL) ** (2 / (gL - 1))
                    p[i] = pL * (c / cL) ** (2 * gL / (gL - 1))
        else:         # right of contact
            if right_fan is None:
                if s >= SR:
                    rho[i], u[i], p[i] = rhoR, uR, pR
                else:
                    rho[i], u[i], p[i] = rho_sR, u_s, p_s
            else:
                if s >= right_fan[1]:
                    rho[i], u[i], p[i] = rhoR, uR, pR
                elif s <= right_fan[0]:
                    rho[i], u[i], p[i] = rho_sR, u_s, p_s
                else:
                    u[i] = 2 / (gR + 1) * (-cR + (gR - 1) / 2 * uR + s)
                    c = 2 / (gR + 1) * (cR - (gR - 1) / 2 * (uR - s))
                    rho[i] = rhoR * (c / cR) ** (2 / (gR - 1))
                    p[i] = pR * (c / cR) ** (2 * gR / (gR - 1))
    return rho, u, p


def exact_field(r, x, t, Ms=MS):
    """Exact (rho, u, p) on a tensor grid x (nx,), t (nt,). Returns arrays (nx, nt)."""
    WL, WR = initial_states(r, Ms)
    star = star_state(WL, WR)
    nx, nt = len(x), len(t)
    R = np.empty((nx, nt)); U = np.empty((nx, nt)); P = np.empty((nx, nt))
    for j, tj in enumerate(t):
        if tj <= 0:
            R[:, j] = np.where(x < X0, WL[0], WR[0])
            U[:, j] = np.where(x < X0, WL[1], WR[1])
            P[:, j] = np.where(x < X0, WL[2], WR[2])
        else:
            R[:, j], U[:, j], P[:, j] = sample((x - X0) / tj, WL, WR, star=star)
    return R, U, P


def wave_summary(r, Ms=MS):
    """Speeds of the three waves, handy for the report table."""
    WL, WR = initial_states(r, Ms)
    p_s, u_s = star_state(WL, WR)
    rhoL, uL, pL = WL; rhoR, uR, pR = WR
    gL, gR = GAMMA_L, GAMMA_R
    cL = np.sqrt(gL * pL / rhoL); cR = np.sqrt(gR * pR / rhoR)
    SR = uR + cR * np.sqrt((gR + 1) / (2 * gR) * p_s / pR + (gR - 1) / (2 * gR))
    if p_s > pL:
        SL = uL - cL * np.sqrt((gL + 1) / (2 * gL) * p_s / pL + (gL - 1) / (2 * gL))
        reflected = "shock"
    else:
        SL = uL - cL
        reflected = "rarefaction"
    return dict(r=r, p_star=p_s, u_star=u_s, reflected=reflected,
                reflected_speed=SL, interface_speed=u_s, transmitted_speed=SR)


if __name__ == "__main__":
    print("post-shock air state (rho, u, p):", post_shock_state())
    for r in [1, 3, 10, 30, 100]:
        print(wave_summary(r))
