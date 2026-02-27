# Reproducibility Scope

This document defines the reproducibility scope and limitations of the SOMBRERO public release.

## What Is Reproducible

### Hot-Start Reference Trajectory Reconstruction

The primary reproducibility target is the **hot-start reconstruction** of the two paper reference trajectories:

1. **JGA case** (Jupiter Gravity Assist):
   - Config: `configs/paper_jga.json`
   - Chromosome: `configs/best_chromosome_jga.json`
   - Command: `python run.py simulate -c configs/paper_jga.json -chr configs/best_chromosome_jga.json`

2. **Direct case** (no JGA):
   - Config: `configs/paper_nojga.json`
   - Chromosome: `configs/best_chromosome_nojga.json`
   - Command: `python run.py simulate -c configs/paper_nojga.json -chr configs/best_chromosome_nojga.json`

**Hot-start** means: running a single trajectory simulation forward-propagation with the provided optimized chromosome (neural network weights). This deterministically reconstructs the reference trajectory from the stored solution, without invoking the evolutionary optimizer.

### Expected Reproduction Accuracy

Scalar outputs are expected to match paper-reported values within approximately 1% relative difference. Sources of variation:

| Source | Impact |
|---|---|
| Integration tolerance (rtol/atol = 1e-5) | ~0.5% on payload mass, ~0.1 yr on t_200AU |
| Chromosome precision (4 significant figures) | ~0.5% on derived quantities |
| Platform/library version differences | ~0.1–0.5% |

Key paper claims (ton-class payload, ~25 yr flight time, specific energy gain) are robust to these variations.

## What Is NOT Claimed

### Independent Rediscovery

The `optimize` mode is provided for completeness and transparency, but this release does **not** claim that running `optimize` from a cold start will independently rediscover the paper reference solutions. Reasons:

- The evolutionary optimizer is stochastic; different seeds, platforms, or library versions may converge to different local optima.
- The paper solutions were found over extended optimization campaigns with specific hyperparameter tuning documented in the paper.
- Warm-start initialization (using provided chromosomes) can be used to refine near the known optimum.

### Bit-Exact Cross-Platform Reproduction

Due to floating-point arithmetic differences across platforms, compilers, and library versions (NumPy, SciPy, PyTorch), bit-exact reproduction is not guaranteed. The integration results are sensitive at the level of integration tolerance.

### Full Optimizer Robustness

No claim is made about optimizer convergence guarantees, statistical robustness of the evolutionary search, or optimality of the found solutions beyond what is documented in the paper.

### Legacy Versions

This release represents a cleaned and refactored version of the codebase used during paper preparation. Historical development versions, intermediate optimization checkpoints, and auxiliary analysis scripts are not included.

## InTrance Verification Artifact

The file `data/paper_artifacts/intrance_verification_fittest_solution.xlsx` contains SOMBRERO's reproduction of the InTrance reference case (Loeb/Ohndorf, r_SOM = 0.7 AU). This is a verification artifact demonstrating that the SOMBRERO trajectory simulator can reproduce published results from an independent optimizer. It is not a paper reference trajectory and is not covered by the hot-start reproduce commands above.

## Sensitivity Analysis

The payload sensitivity analysis presented in the paper (payload maps vs. EPS specific power and structural mass fraction) is derived from closed-form scaling relations applied to the hot-start reference trajectory outputs. These relations are documented in the paper (Eq. references in manuscript). The sensitivity maps are reproducible from the scalar outputs of the two hot-start reconstructions.
