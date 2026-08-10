# Data Artifacts

Excel exports from the original optimization campaigns. They are cross-check artifacts, not the authoritative source of paper results: reproduce the trajectories with `run.py simulate` and the provided configs and chromosomes.

| Curated Name | Original Name | Case |
|---|---|---|
| `jga_fittest_solution.xlsx` | `7.1_JGA_fittest_solution.xlsx` | JGA reference trajectory (paper Table 5) |
| `nojga_fittest_solution.xlsx` | `5.5_noJGA_fittest_solution.xlsx` | Direct reference trajectory (paper Table 6) |

Notes:

- Scalar columns (payload mass, flight time, fitness score, power) are the campaigns' own outputs and correspond to the paper quantities (see the JGA payload note below). They record the campaign settings (`rtol = atol = 1e-5`) and the campaign EA hyperparameters (`mutation_rate = 0.1`; `mutation_std_dev = 0.1` for the JGA campaign, `1` for the direct campaign), which differ from the example values in the shipped configs; re-simulating with the shipped configs reproduces them within the ~1% band (see `docs/reproducibility_scope.md`).
- The chromosome stored in each artifact is identical to the corresponding `configs/best_chromosome_*.json` (the JSON chromosomes were transcribed from these exports).
- Re-simulating the published 4-significant-figure chromosomes reproduces the direct case to four significant figures. For the JGA case this artifact records 3,110.7 kg and re-simulation yields about 3,110 kg, against the 3,083 kg reported in the manuscript: a 0.9% difference, inside the ~1% band.
- Array-valued columns (state vectors, chromosome) may appear as stringified lists in single cells, an artifact of the `pandas.DataFrame.to_excel()` export; array values are fixed-format strings with 4 significant figures (numpy print options), while scalar columns are full precision.
- The original file names encode the approximate fitness score at export.
