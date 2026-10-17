"""
PINN for the 1-D shock / material-interface problem.

Unknowns: rho(x,t), u(x,t), p(x,t). PDE: 1-D Euler in conservation form
    d/dt [rho, rho u, E] + d/dx [rho u, rho u^2 + p, u (E + p)] = nu * d2/dx2 [...]
with E = p/(gamma-1) + rho u^2 / 2 and nu = 0 unless we switch on artificial
viscosity. IC: the Riemann data. BC: Dirichlet with the initial states at
x = 0 and x = 1 (no wave reaches the boundaries before T_END).

Everything that differs between experiments is a field in Config, so each
experiment script is just a list of configs.
"""
import json
import math
import os
import time
from dataclasses import dataclass, asdict, field

import numpy as np
import torch

from riemann import GAMMA_L, X0, T_END, initial_states, exact_field

torch.set_num_threads(1)   # we run two jobs side by side on a 2-core machine
G = GAMMA_L


@dataclass
class Config:
    name: str = "vanilla"
    r: float = 10.0                 # difficulty knob: density ratio heavy/light
    seed: int = 0
    width: int = 64
    depth: int = 4
    n_res: int = 4000               # collocation points
    n_ic: int = 400
    n_bc: int = 200
    iters: int = 12000
    lr: float = 1e-3
    mode: str = "pinn"              # "pinn" or "supervised" (fit exact solution directly)
    normalize: bool = False         # scale each residual / IC term by its variable's size
    nu: float = 0.0                 # artificial viscosity (constant)
    nu_final: float = None          # if set, decay nu -> nu_final over training (curriculum)
    adaptive: bool = False          # residual-based adaptive resampling (RAD)
    logrho: bool = False            # network predicts log(rho) instead of rho
    causal: float = 0.0             # causal weighting tolerance eps (Wang et al. 2024); 0 = off
    n_tbins: int = 16
    adapt_every: int = 1000
    log_every: int = 100
    tag: str = ""


class MLP(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        layers, d_in = [], 2
        for _ in range(depth):
            layers += [torch.nn.Linear(d_in, width), torch.nn.Tanh()]
            d_in = width
        layers += [torch.nn.Linear(d_in, 3)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x, t):
        # map x in [0,1], t in [0,T] to [-1,1]
        z = torch.cat([2 * x - 1, 2 * t / T_END - 1], dim=1)
        return self.net(z)


class Problem:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        WL, WR = initial_states(cfg.r)
        self.WL, self.WR = WL, WR
        # characteristic scales (used for output scaling and, if asked, for loss normalization)
        self.rho_s = max(WL[0], WR[0])
        self.u_s = max(abs(WL[1]), 1.0)
        self.p_s = max(WL[2], WR[2])
        # for the log-density option: centre and spread of log(rho) over the expected range
        lo, hi = min(WL[0], WR[0]), 3.0 * max(WL[0], WR[0])
        self.lr_c = 0.5 * math.log(lo * hi)
        self.lr_s = max(0.5 * math.log(hi / lo), 0.5)

    def prim(self, out):
        if self.cfg.logrho:
            rho = torch.exp(self.lr_c + self.lr_s * out[:, :1])
        else:
            rho = self.rho_s * out[:, :1]
        return rho, self.u_s * out[:, 1:2], self.p_s * out[:, 2:3]

    def ic(self, x):
        WL, WR = self.WL, self.WR
        left = (x < X0).float()
        rho = left * WL[0] + (1 - left) * WR[0]
        u = left * WL[1] + (1 - left) * WR[1]
        p = left * WL[2] + (1 - left) * WR[2]
        return rho, u, p


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def residuals(model, prob, x, t, nu):
    rho, u, p = prob.prim(model(x, t))
    m = rho * u
    E = p / (G - 1) + 0.5 * rho * u**2
    U = [rho, m, E]
    F = [m, m * u + p, u * (E + p)]
    res = []
    for Uk, Fk in zip(U, F):
        r_k = grad(Uk, t) + grad(Fk, x)
        if nu > 0:
            r_k = r_k - nu * grad(grad(Uk, x), x)
        res.append(r_k)
    return res


def train(cfg: Config, outdir="../results/runs", verbose=True):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    prob = Problem(cfg)
    model = MLP(cfg.width, cfg.depth)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.1 ** (1 / cfg.iters))  # lr: 1e-3 -> 1e-4

    # training points
    def rand_res(n):
        return torch.rand(n, 1), T_END * torch.rand(n, 1)
    x_r, t_r = rand_res(cfg.n_res)
    x_0 = torch.rand(cfg.n_ic, 1)
    t_b = T_END * torch.rand(cfg.n_bc, 1)
    rho0, u0, p0 = prob.ic(x_0)
    bcL = [torch.full_like(t_b, v) for v in prob.WL]
    bcR = [torch.full_like(t_b, v) for v in prob.WR]

    # scales for the "normalize" option: residual k is divided by a typical size of that equation
    sc_var = [prob.rho_s, prob.u_s, prob.p_s]
    c_ref = math.sqrt(G * prob.p_s / min(prob.WL[0], prob.WR[0]))   # fastest sound speed
    sc_res = [prob.rho_s * c_ref, prob.rho_s * c_ref**2, prob.p_s * c_ref]
    if not cfg.normalize:
        sc_var = [1.0, 1.0, 1.0]
        sc_res = [1.0, 1.0, 1.0]

    # supervised mode: exact solution at the collocation points
    if cfg.mode == "supervised":
        def exact_at(x, t):
            xs, ts = x.numpy().ravel(), t.numpy().ravel()
            from riemann import sample, star_state
            st = star_state(prob.WL, prob.WR)
            out = np.stack(sample((xs - X0) / np.maximum(ts, 1e-12), prob.WL, prob.WR, star=st), 1)
            return torch.tensor(out, dtype=torch.float32)
        y_r = exact_at(x_r, t_r)

    hist = {k: [] for k in ["it", "pde_mass", "pde_mom", "pde_energy", "ic", "bc", "total", "err_rho", "wmin"]}
    xe = np.linspace(0, 1, 201); te = np.linspace(0, T_END, 61)
    Re, Ue, Pe = exact_field(cfg.r, xe, te)
    Xg, Tg = np.meshgrid(xe, te, indexing="ij")
    Xt = torch.tensor(Xg.reshape(-1, 1), dtype=torch.float32)
    Tt = torch.tensor(Tg.reshape(-1, 1), dtype=torch.float32)

    def evaluate():
        with torch.no_grad():
            rho, u, p = prob.prim(model(Xt, Tt))
        R = rho.numpy().reshape(Xg.shape); U = u.numpy().reshape(Xg.shape); P = p.numpy().reshape(Xg.shape)
        return R, U, P

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    t0 = time.perf_counter()
    for it in range(cfg.iters + 1):
        if cfg.nu_final is not None:
            frac = min(1.0, it / (0.8 * cfg.iters))
            nu = cfg.nu * (cfg.nu_final / cfg.nu) ** frac   # geometric decay, then hold
        else:
            nu = cfg.nu

        if cfg.adaptive and it > 0 and it % cfg.adapt_every == 0 and it < 0.9 * cfg.iters:
            # RAD: draw a big candidate pool, keep points with probability ~ |residual|
            xc, tc = rand_res(20 * cfg.n_res)
            xc.requires_grad_(); tc.requires_grad_()
            rc = residuals(model, prob, xc, tc, nu)
            score = sum((rk / s).pow(2) for rk, s in zip(rc, sc_res)).detach().sqrt().ravel()
            prob_w = score / score.mean() + 1.0          # +1 keeps some uniform coverage
            idx = torch.multinomial(prob_w / prob_w.sum(), cfg.n_res, replacement=False)
            x_r, t_r = xc.detach()[idx], tc.detach()[idx]

        opt.zero_grad()
        if cfg.mode == "supervised":
            out = torch.cat(prob.prim(model(x_r, t_r)), 1)
            loss = (((out - y_r) / torch.tensor([prob.rho_s, prob.u_s, prob.p_s])) ** 2).mean()
            parts = dict(pde_mass=0.0, pde_mom=0.0, pde_energy=0.0, ic=0.0, bc=0.0, wmin=1.0)
        else:
            xr = x_r.clone().requires_grad_(); tr = t_r.clone().requires_grad_()
            res = residuals(model, prob, xr, tr, nu)
            if cfg.causal > 0:
                # split the collocation points into time bins; weight bin i by exp(-eps * sum of earlier bin losses)
                per_pt = [((rk / s) ** 2).ravel() for rk, s in zip(res, sc_res)]
                tot = sum(per_pt)
                b = torch.clamp((tr.detach().ravel() / T_END * cfg.n_tbins).long(), max=cfg.n_tbins - 1)
                cnt = torch.bincount(b, minlength=cfg.n_tbins).clamp(min=1).float()
                Lb = torch.zeros(cfg.n_tbins).index_add(0, b, tot) / cnt
                w = torch.exp(-cfg.causal * torch.cumsum(Lb.detach(), 0) + cfg.causal * Lb.detach())
                wpt = w[b]
                L_pde = [(wpt * pp).sum() / cfg.n_res for pp in per_pt]
                causal_wmin = w.min().item()
            else:
                L_pde = [((rk / s) ** 2).mean() for rk, s in zip(res, sc_res)]
            rho_i, u_i, p_i = prob.prim(model(x_0, torch.zeros_like(x_0)))
            L_ic = sum((((a - b) / s) ** 2).mean() for a, b, s in zip((rho_i, u_i, p_i), (rho0, u0, p0), sc_var))
            pl = prob.prim(model(torch.zeros_like(t_b), t_b))
            pr = prob.prim(model(torch.ones_like(t_b), t_b))
            L_bc = sum((((a - b) / s) ** 2).mean() for a, b, s in zip(pl, bcL, sc_var)) + \
                   sum((((a - b) / s) ** 2).mean() for a, b, s in zip(pr, bcR, sc_var))
            loss = sum(L_pde) + L_ic + L_bc
            parts = dict(pde_mass=L_pde[0].item(), pde_mom=L_pde[1].item(),
                         pde_energy=L_pde[2].item(), ic=L_ic.item(), bc=L_bc.item(),
                         wmin=causal_wmin if cfg.causal > 0 else 1.0)
        loss.backward()
        opt.step()
        sched.step()

        if it % cfg.log_every == 0:
            hist["it"].append(it)
            for k, v in parts.items():
                hist[k].append(v)
            hist["total"].append(loss.item())
            if it % (cfg.log_every * 10) == 0:
                R, _, _ = evaluate()
                hist["err_rho"].append(rel(R, Re))
                if verbose:
                    print(f"[{cfg.name} r={cfg.r:g}] it {it:6d} loss {loss.item():.3e} "
                          f"pde {parts['pde_mass']:.1e}/{parts['pde_mom']:.1e}/{parts['pde_energy']:.1e} "
                          f"ic {parts['ic']:.1e} bc {parts['bc']:.1e} wmin {parts['wmin']:.2f} err_rho {hist['err_rho'][-1]:.3f}",
                          flush=True)
            else:
                hist["err_rho"].append(None)
    train_time = time.perf_counter() - t0

    # inference cost on the full space-time grid
    t1 = time.perf_counter()
    for _ in range(10):
        R, U, P = evaluate()
    infer_time = (time.perf_counter() - t1) / 10

    result = dict(cfg=asdict(cfg), train_time=train_time, infer_time=infer_time,
                  err_rho=rel(R, Re), err_u=rel(U, Ue), err_p=rel(P, Pe),
                  err_rho_final_t=rel(R[:, -1], Re[:, -1]), hist=hist)
    os.makedirs(outdir, exist_ok=True)
    stem = f"{cfg.name}_r{cfg.r:g}_s{cfg.seed}{cfg.tag}"
    json.dump(result, open(os.path.join(outdir, stem + ".json"), "w"))
    np.savez_compressed(os.path.join(outdir, stem + ".npz"), x=xe, t=te, R=R, U=U, P=P,
                        Re=Re, Ue=Ue, Pe=Pe, x_r=x_r.numpy(), t_r=t_r.numpy())
    if verbose:
        print(f"[{cfg.name} r={cfg.r:g}] done: err rho {result['err_rho']:.3f} u {result['err_u']:.3f} "
              f"p {result['err_p']:.3f}  train {train_time:.0f}s", flush=True)
    return result


if __name__ == "__main__":
    import sys
    cfg = Config(r=float(sys.argv[1]) if len(sys.argv) > 1 else 1.0, iters=2000, name="smoke")
    train(cfg, outdir="../results/smoke")
