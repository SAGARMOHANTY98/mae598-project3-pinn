# Project 3: Can a PINN handle a shock hitting a material interface?

MAE 598 Design Optimization, Project 3 (PINN/PINO in the Field)
Team: [names]
Researcher: Prof. Marcus Herrmann, SEMTE, ASU

**The report is [`report.ipynb`](report.ipynb).** It contains the outreach log, problem brief, minimal case formulation, all experiments (baseline, loss plots, error vs. knob, error maps, cost, diagnosis), the recommendation memo, and limitations. All cells have been run, so it reads fine directly on GitHub.

## Short version

Following Prof. Herrmann's advice, we tested a minimal 1-D version of a planar shock hitting a drop: a Mach 1.5 shock reaching a single interface between a light and a heavy gas, with the density ratio $r$ (1 to 100) as the difficulty knob. We compared a PINN against the exact Riemann solution and a simple finite-volume solver.

- The PINN's density error goes from 8% at $r = 1$ to 36% at $r = 10$, and at $r = 100$ it misses the shocked heavy layer completely. A 100-cell finite-volume run (0.03 s) matches or beats every PINN variant; each PINN takes 10–25 min to train.
- The same network fit directly to the exact solution is accurate (1–8%), so the problem is the physics-informed loss, not the network.
- Diagnosis (Path B): a loss imbalance that grows with $r$ (fixed by scaling), on top of a strong-form residual that cannot represent jumps. Adaptive sampling, a $\log\rho$ output and causal weighting did not help.
- Partial fix (Path C): an artificial-viscosity curriculum cuts the shock-only error from 8% to 3%, but cannot save the interface, because the viscosity needed smears a contact about as wide as the dense layer itself.

## Running it

```bash
pip install -r requirements.txt     # numpy, scipy, matplotlib, torch (CPU is enough)
jupyter notebook report.ipynb       # run all cells
```

With the default settings the notebook rebuilds the baseline, every table and every figure from the stored runs in `results/runs/` (a couple of minutes), then retrains one PINN from scratch with a fixed seed to check it matches the stored result (about 10 min). Set `RUN_ALL_PINNS = True` in the first code cell to retrain everything (about 5 CPU-hours). Section 7 of the notebook has the command-line equivalents.

## Layout

| path | contents |
|---|---|
| `report.ipynb` | the report |
| `src/` | exact Riemann solver, finite-volume baseline, PINN, experiment runner, figure code |
| `results/runs/` | one `.json` (config, loss history, errors, timings) and one `.npz` (predicted fields) per PINN run |
| `results/*.json, *.csv` | baseline errors, summary table, interface metrics |
| `figures/` | all figures used in the report |
