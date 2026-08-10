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

Scalar outputs are expected to match paper-reported values within approximately 1% relative difference. Measured integration-tolerance sensitivity of the reference cases is small (below 0.05% across `rtol = atol = 1e-5` to `1e-10`); the band is dominated by the 4-significant-figure chromosome transcription and platform/library differences relative to the original campaign environment.

Key paper claims (ton-class payload, ~25 yr flight time, specific energy gain) are insensitive to these variations.

### Model Simplifications

The dynamics are two-dimensional and heliocentric. Jupiter is not propagated as a body: the gravity assist is applied analytically when the trajectory crosses r = 5.2 AU, assuming a circular, coplanar Jupiter orbit (orbital speed 13.06 km/s), favourable phasing, and a fixed assist periapsis of 1.34 Jupiter radii (equatorial radius 71,492 km). Launch-window phasing is therefore not constrained; as stated in the paper, the JGA contribution is computed under favourable Earth–Jupiter phasing and is an optimistic estimate.

## What Is Not Claimed

### Independent Rediscovery

The `optimize` mode is provided for completeness and transparency, but this release does not claim that running `optimize` from a cold start will independently rediscover the paper reference solutions. Reasons:

- The evolutionary optimizer is stochastic; different seeds, platforms, or library versions may converge to different local optima.
- The paper solutions were found over extended optimization campaigns; the campaign hyperparameters are recorded in the artifacts under `data/paper_artifacts/` and differ from the example values in the shipped configs.
- Warm-start initialization can be used to refine near a known optimum: set `init.init_mode: "warm"` in the config to seed the population with the provided chromosome.

### Bit-Exact Cross-Platform Reproduction

Due to floating-point arithmetic differences across platforms, compilers, and library versions (NumPy, SciPy, PyTorch), bit-exact reproduction is not guaranteed. The integration results are sensitive at the level of integration tolerance.

### Optimizer Guarantees

No claim is made about optimizer convergence guarantees, statistical robustness of the evolutionary search, or optimality of the found solutions beyond what is documented in the paper.

### Legacy Versions

This release represents a cleaned and refactored version of the codebase used during paper preparation. Historical development versions, intermediate optimization checkpoints, and auxiliary analysis scripts are not included.

## Sensitivity Analysis

The payload sensitivity analysis presented in the paper (payload maps vs. EPS specific power and structural mass fraction) is derived from closed-form scaling relations applied to the hot-start reference trajectory outputs. These relations are documented in the paper. The sensitivity maps are reproducible from the scalar outputs of the two hot-start reconstructions.
