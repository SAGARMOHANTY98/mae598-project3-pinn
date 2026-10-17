"""Make every figure in the README from results/ . Run after the experiments."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from riemann import exact_field, T_END, X0
from fv_solver import solve

RUNS = "../results/runs"
FIG = "../figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})
KNOB = [1, 3, 10, 30, 100]
COL = {"vanilla": "C3", "supervised": "C7", "normalized": "C1", "rad": "C4", "visc": "C2",
       "causal": "C5", "logrho": "C9", "fix": "C0"}
LABEL = {"vanilla": "PINN, vanilla", "supervised": "same net, supervised fit",
         "normalized": "PINN, normalized loss", "rad": "normalized + adaptive sampling",
         "visc": "normalized + viscosity", "causal": "normalized + causal weights",
         "logrho": r"normalized + $\log\rho$ output", "fix": r"normalized + $\log\rho$ + viscosity"}


def load(name, r, seed=0):
    f = f"{RUNS}/{name}_r{r:g}_s{seed}"
    if not os.path.exists(f + ".json"):
        return None, None
    return json.load(open(f + ".json")), np.load(f + ".npz")


def all_seeds(name, r):
    out = []
    for f in sorted(glob.glob(f"{RUNS}/{name}_r{r:g}_s*.json")):
        out.append(json.load(open(f)))
    return out


# ---------------------------------------------------------------- fig 1: the problem
def fig_problem():
    x = np.linspace(0, 1, 801)
    t = np.linspace(0, T_END, 241)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    R, _, _ = exact_field(10, x, t)
    im = ax[0].pcolormesh(x, t, np.log10(R.T), shading="auto", cmap="viridis")
    plt.colorbar(im, ax=ax[0], label=r"$\log_{10}\rho$")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("t"); ax[0].set_title("exact density, r = 10")
    ax[0].grid(False)
    for r, c in zip([1, 10, 100], ["C0", "C1", "C3"]):
        R, U, P = exact_field(r, x, np.array([T_END]))
        ax[1].semilogy(x, R[:, 0], c, label=f"r = {r}")
        ax[2].plot(x, P[:, 0], c, label=f"r = {r}")
    ax[1].set_title(r"density at $t = T$"); ax[1].set_xlabel("x"); ax[1].legend()
    ax[2].set_title(r"pressure at $t = T$"); ax[2].set_xlabel("x")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_problem.png"); plt.close(fig)


# ---------------------------------------------------------------- fig 2: P1 baseline
def fig_baseline():
    d = json.load(open("../results/fv_baseline.json"))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for i, r in enumerate(KNOB):
        nx = sorted(int(k) for k in d[str(r)])
        e = [d[str(r)][str(n)]["err_rho"] for n in nx]
        w = [d[str(r)][str(n)]["wall"] for n in nx]
        ax[0].loglog(nx, e, "o-", color=f"C{i}", label=f"r = {r}")
        ax[1].loglog(w, e, "o-", color=f"C{i}")
    n = np.array([100, 1600]); ax[0].loglog(n, 0.3 * (n / 100) ** -0.5, "k--", lw=0.8, label=r"$N^{-1/2}$")
    ax[0].set_xlabel("cells"); ax[0].set_ylabel(r"rel. $L_2$ error in $\rho$ at $t=T$"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("wall-clock [s]"); ax[1].set_ylabel(r"rel. $L_2$ error in $\rho$")
    ax[0].set_title("P1: finite-volume baseline"); ax[1].set_title("accuracy vs cost")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_baseline.png"); plt.close(fig)


# ---------------------------------------------------------------- fig 3: P2 loss histories
def fig_losses(names=("vanilla", "normalized"), rs=(1, 100)):
    fig, axs = plt.subplots(len(names), len(rs), figsize=(5 * len(rs), 3.2 * len(names)), squeeze=False)
    for i, nm in enumerate(names):
        for j, r in enumerate(rs):
            res, _ = load(nm, r)
            ax = axs[i, j]
            if res is None:
                ax.set_visible(False); continue
            h = res["hist"]
            for k, c in zip(["pde_mass", "pde_mom", "pde_energy", "ic", "bc"], ["C0", "C1", "C2", "C3", "C4"]):
                ax.semilogy(h["it"], h[k], c, lw=1, label=k)
            ax2 = ax.twinx()
            it = [a for a, e in zip(h["it"], h["err_rho"]) if e is not None]
            er = [e for e in h["err_rho"] if e is not None]
            ax2.plot(it, er, "k--", lw=1.2, label=r"$\rho$ error")
            ax2.set_ylim(0, max(1.0, max(er) * 1.05)); ax2.grid(False)
            ax2.set_ylabel(r"rel. $L_2$ error in $\rho$ (dashed)")
            ax.set_title(f"{LABEL[nm]}, r = {r}"); ax.set_xlabel("iteration")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_losses.png"); plt.close(fig)


# ---------------------------------------------------------------- fig 4: P3 error vs knob
def fig_knob(names):
    fv = json.load(open("../results/fv_baseline.json"))
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for nm in names:
        rs, mean, lo, hi = [], [], [], []
        for r in KNOB:
            runs = all_seeds(nm, r)
            if not runs:
                continue
            e = np.array([q["err_rho"] for q in runs])
            rs.append(r); mean.append(e.mean()); lo.append(e.min()); hi.append(e.max())
        if rs:
            ax.plot(rs, mean, "o-", color=COL[nm], label=LABEL[nm])
            ax.fill_between(rs, lo, hi, color=COL[nm], alpha=0.15)
    fvst = json.load(open("../results/fv_baseline_spacetime.json"))
    e400 = [fvst[str(r)]["400"]["err_rho"] for r in KNOB]
    ax.plot(KNOB, e400, "ks--", label="finite volume, 400 cells (0.2 s)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"density ratio $r = \rho_{heavy}/\rho_{light}$  (difficulty knob)")
    ax.set_ylabel(r"rel. $L_2$ error in $\rho$ over $(x,t)$")
    ax.set_title("P3: error vs knob")
    ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_knob.png"); plt.close(fig)


# ---------------------------------------------------------------- fig 5: P4 where is the error
def fig_where(name="vanilla", rs=(1, 10, 100), tag="fig5_where"):
    fig, axs = plt.subplots(2, len(rs), figsize=(4.2 * len(rs), 6.4))
    for j, r in enumerate(rs):
        res, d = load(name, r)
        if res is None:
            continue
        err = np.abs(d["R"] - d["Re"]) / d["Re"].max()
        im = axs[0, j].pcolormesh(d["x"], d["t"], err.T, shading="auto", cmap="magma", vmin=0, vmax=0.5)
        axs[0, j].set_title(f"{LABEL[name]}, r = {r}\n" + r"$|\rho-\rho_{exact}|/\max\rho$")
        axs[0, j].set_xlabel("x"); axs[0, j].set_ylabel("t"); axs[0, j].grid(False)
        plt.colorbar(im, ax=axs[0, j])
        # error spectrum at the final time vs spectrum of the solution itself
        e = d["R"][:, -1] - d["Re"][:, -1]
        k = np.fft.rfftfreq(len(e), d=d["x"][1] - d["x"][0])
        axs[1, j].semilogy(k[1:], np.abs(np.fft.rfft(e))[1:], "C3", label="error")
        axs[1, j].semilogy(k[1:], np.abs(np.fft.rfft(d["Re"][:, -1] - d["Re"][:, -1].mean()))[1:], "k", lw=0.8,
                           label="exact solution")
        axs[1, j].set_xlabel("wavenumber (cycles per unit length)")
        axs[1, j].set_title(r"Fourier spectrum at $t=T$"); axs[1, j].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/{tag}.png"); plt.close(fig)


# ---------------------------------------------------------------- fig 6: profiles at final time
def fig_profiles(names, rs=(1, 10, 100), tag="fig6_profiles"):
    fig, axs = plt.subplots(1, len(rs), figsize=(4.4 * len(rs), 3.6))
    for j, r in enumerate(rs):
        ax = axs[j]
        x = np.linspace(0, 1, 801)
        R, _, _ = exact_field(r, x, np.array([T_END]))
        ax.plot(x, R[:, 0], "k", lw=2, label="exact")
        xf, (rf, _, _), _ = solve(r, 400)
        ax.plot(xf, rf, "k:", lw=1.2, label="FV 400 cells")
        for nm in names:
            res, d = load(nm, r)
            if res is None:
                continue
            ax.plot(d["x"], d["R"][:, -1], color=COL[nm], lw=1.3, label=LABEL[nm])
        ax.set_title(f"density at t = T, r = {r}"); ax.set_xlabel("x")
        if j == 1:
            ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    fig.tight_layout(); fig.savefig(f"{FIG}/{tag}.png"); plt.close(fig)


# ---------------------------------------------------------------- table
def table(names):
    rows = []
    for nm in names:
        for r in KNOB:
            for q in all_seeds(nm, r):
                rows.append((nm, r, q["cfg"]["seed"], q["err_rho"], q["err_u"], q["err_p"],
                             q["train_time"], q["infer_time"]))
    with open("../results/summary.csv", "w") as f:
        f.write("experiment,r,seed,err_rho,err_u,err_p,train_s,infer_s\n")
        for row in rows:
            f.write("%s,%g,%d,%.4f,%.4f,%.4f,%.1f,%.4f\n" % row)
    return rows


if __name__ == "__main__":
    names = [n for n in COL if glob.glob(f"{RUNS}/{n}_*.json")]
    fig_problem(); fig_baseline()
    fig_losses(("vanilla", "normalized", "visc"))
    fig_knob(names)
    fig_where("vanilla")
    fig_where("visc", tag="fig7_where_visc")
    fig_profiles(["vanilla", "normalized", "visc"], tag="fig6_profiles")
    fig_profiles(["rad", "logrho", "causal"], tag="fig8_profiles_other")
    for row in table(names):
        print(row)


# ---------------------------------------------------------------- interface metrics
def interface_table(names, rs=(10, 100)):
    """Where does each model put the interface at t = T, and how dense is its shocked heavy layer?
    Interface = first x (from the left) where rho exceeds the geometric mean of the two star densities."""
    from riemann import initial_states, star_state
    rows = []
    for r in rs:
        WL, WR = initial_states(r)
        p_s, u_s = star_state(WL, WR)
        x = np.linspace(0, 1, 2001)
        R, _, _ = exact_field(r, x, np.array([T_END]))
        rho_layer = R[:, 0].max()
        rhoL_star = R[(x > X0 - 0.05) & (x < X0 + u_s * T_END - 0.005), 0].min()
        thr = np.sqrt(rho_layer * rhoL_star)
        x_exact = X0 + u_s * T_END
        for nm in ["fv"] + list(names):
            if nm == "fv":
                xx, (rho, _, _), _ = solve(r, 400)
            else:
                res, d = load(nm, r)
                if res is None:
                    continue
                xx, rho = d["x"], d["R"][:, -1]
            idx = np.where((rho > thr) & (xx > 0.2))[0]
            x_int = xx[idx[0]] if len(idx) else np.nan
            rows.append((nm, r, x_exact, x_int, x_int - x_exact, rho.max() / rho_layer))
    with open("../results/interface.csv", "w") as f:
        f.write("model,r,x_interface_exact,x_interface_model,position_error,peak_density_fraction\n")
        for row in rows:
            f.write("%s,%g,%.4f,%.4f,%.4f,%.3f\n" % row)
    return rows


if __name__ == "__main__":
    for row in interface_table([n for n in COL if glob.glob(f"{RUNS}/{n}_*.json")]):
        print(row)
