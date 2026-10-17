"""
P1 baseline: a standard second-order finite-volume solver for the 1-D Euler
equations (MUSCL reconstruction on primitive variables with a minmod limiter,
HLLC flux, SSP-RK2 time stepping, transmissive boundaries).

This is the kind of thing a CFD group would write in an afternoon, which is the
point: it is the bar the PINN has to clear.
"""
import time
import numpy as np
from riemann import GAMMA_L, X0, T_END, initial_states, exact_field

G = GAMMA_L


def prim2cons(rho, u, p):
    return np.stack([rho, rho * u, p / (G - 1) + 0.5 * rho * u**2])


def cons2prim(U):
    rho = U[0]
    u = U[1] / rho
    p = (G - 1) * (U[2] - 0.5 * rho * u**2)
    return rho, u, p


def flux(rho, u, p):
    E = p / (G - 1) + 0.5 * rho * u**2
    return np.stack([rho * u, rho * u**2 + p, u * (E + p)])


def hllc(WL, WR):
    rL, uL, pL = WL
    rR, uR, pR = WR
    cL = np.sqrt(G * pL / rL); cR = np.sqrt(G * pR / rR)
    SL = np.minimum(uL - cL, uR - cR)
    SR = np.maximum(uL + cL, uR + cR)
    SM = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / (rL * (SL - uL) - rR * (SR - uR))
    UL = prim2cons(rL, uL, pL); UR = prim2cons(rR, uR, pR)
    FL = flux(rL, uL, pL); FR = flux(rR, uR, pR)

    def ustar(r, u, p, S, U):
        fac = r * (S - u) / (S - SM)
        E = U[2]
        return np.stack([fac, fac * SM, fac * (E / r + (SM - u) * (SM + p / (r * (S - u))))])

    UsL = ustar(rL, uL, pL, SL, UL)
    UsR = ustar(rR, uR, pR, SR, UR)
    F = np.where(SL >= 0, FL,
        np.where(SM >= 0, FL + SL * (UsL - UL),
        np.where(SR > 0, FR + SR * (UsR - UR), FR)))
    return F


def minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def rhs(U, dx):
    # two ghost cells each side, zero-gradient
    Ug = np.concatenate([U[:, :1], U[:, :1], U, U[:, -1:], U[:, -1:]], axis=1)
    W = np.stack(cons2prim(Ug))
    dW = minmod(W[:, 1:-1] - W[:, :-2], W[:, 2:] - W[:, 1:-1])     # slopes for cells 1..n+2
    WLf = W[:, 1:-2] + 0.5 * dW[:, :-1]    # left state at faces
    WRf = W[:, 2:-1] - 0.5 * dW[:, 1:]     # right state at faces
    F = hllc(WLf, WRf)                      # n+1 faces
    return -(F[:, 1:] - F[:, :-1]) / dx


def solve(r, nx=400, cfl=0.5, t_end=T_END, save_times=None):
    x = (np.arange(nx) + 0.5) / nx
    dx = 1.0 / nx
    WL, WR = initial_states(r)
    rho = np.where(x < X0, WL[0], WR[0])
    u = np.where(x < X0, WL[1], WR[1])
    p = np.where(x < X0, WL[2], WR[2])
    U = prim2cons(rho, u, p)
    t = 0.0
    snaps = {}
    save_times = [] if save_times is None else list(save_times)
    k = 0
    while t < t_end - 1e-14:
        rho, u, p = cons2prim(U)
        smax = np.max(np.abs(u) + np.sqrt(G * p / rho))
        dt = min(cfl * dx / smax, t_end - t)
        if k < len(save_times) and t + dt > save_times[k]:
            dt = save_times[k] - t
        U1 = U + dt * rhs(U, dx)
        U = 0.5 * (U + U1 + dt * rhs(U1, dx))
        t += dt
        while k < len(save_times) and abs(t - save_times[k]) < 1e-12:
            snaps[save_times[k]] = cons2prim(U)
            k += 1
    return x, cons2prim(U), snaps


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def spacetime_error(r, nx):
    """FV error measured exactly like the PINN error: rel. L2 over the same 201 x 61 (x, t) grid."""
    xe = np.linspace(0, 1, 201); te = np.linspace(0, T_END, 61)
    t0 = time.perf_counter()
    x, _, snaps = solve(r, nx, save_times=te[1:])
    wall = time.perf_counter() - t0
    Re, Ue, Pe = exact_field(r, xe, te)
    WL, WR = initial_states(r)
    R = np.empty_like(Re); U = np.empty_like(Re); P = np.empty_like(Re)
    R[:, 0], U[:, 0], P[:, 0] = Re[:, 0], Ue[:, 0], Pe[:, 0]
    for j, tj in enumerate(te[1:], start=1):
        rho, u, p = snaps[tj]
        R[:, j] = np.interp(xe, x, rho); U[:, j] = np.interp(xe, x, u); P[:, j] = np.interp(xe, x, p)
    return dict(wall=wall, err_rho=rel_l2(R, Re), err_u=rel_l2(U, Ue), err_p=rel_l2(P, Pe))


if __name__ == "__main__":
    import json, os
    st = {}
    for r in [1, 3, 10, 30, 100]:
        st[r] = {nx: spacetime_error(r, nx) for nx in [100, 400, 1600]}
        print("space-time", r, {nx: round(v["err_rho"], 4) for nx, v in st[r].items()})
    os.makedirs("../results", exist_ok=True)
    json.dump(st, open("../results/fv_baseline_spacetime.json", "w"), indent=1)
    out = {}
    for r in [1, 3, 10, 30, 100]:
        out[r] = {}
        for nx in [100, 200, 400, 800, 1600]:
            t0 = time.perf_counter()
            x, (rho, u, p), _ = solve(r, nx)
            wall = time.perf_counter() - t0
            Re, Ue, Pe = exact_field(r, x, np.array([T_END]))
            out[r][nx] = dict(wall=wall, err_rho=rel_l2(rho, Re[:, 0]),
                              err_u=rel_l2(u, Ue[:, 0]), err_p=rel_l2(p, Pe[:, 0]))
            print(r, nx, out[r][nx])
    os.makedirs("../results", exist_ok=True)
    json.dump(out, open("../results/fv_baseline.json", "w"), indent=1)
