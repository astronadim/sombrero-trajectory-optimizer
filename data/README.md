# Data Artifacts

This directory contains curated Excel artifacts from the paper production process.
These are **supporting cross-check artifacts (summary exports from the original optimization campaigns)**, not the authoritative source of paper results.

**Primary reproducibility is via code + configs + chromosomes + `simulate` mode.**
Reviewers should use `run.py simulate` with the provided configs and chromosomes as the definitive reproduction pathway.

## File Inventory

| Curated Name | Original Name | Case | Description |
|---|---|---|---|
| `jga_fittest_solution.xlsx` | `7.05_JGA_fittest_solution_2025.xlsx` | JGA | Trajectory time-series and scalar summary for the optimized JGA reference trajectory |
| `nojga_fittest_solution.xlsx` | `5.5_noJGA_fittest_solution.xlsx` | Direct (no JGA) | Trajectory time-series and scalar summary for the optimized direct reference trajectory |
| `intrance_verification_fittest_solution.xlsx` | `verification_InTrance_fittest_solution.xlsx` | InTrance verification | Verification artifact: SOMBRERO reproduction of the InTrance reference case (Loeb/Ohndorf) |

## Notes

- **Relation to paper claims:** The JGA and noJGA artifacts correspond to the paper's Table 3 and Table 4 reference trajectories, respectively. The InTrance artifact corresponds to the verification/validation comparison discussed in the paper's verification section.
- **Data format caveat:** Some array-valued columns (e.g., trajectory state vectors, chromosome) may be stored as stringified Python lists within single Excel cells. This is a serialization artifact from the `pandas.DataFrame.to_excel()` export used during optimization campaigns.
- **Scalar values:** Key scalar metrics (payload mass, flight time, fitness score, power) are stored in dedicated columns and are directly comparable to paper-reported values.
- **Fitness score naming:** The original file names (e.g., `7.05_JGA_...`, `5.5_noJGA_...`) encode the approximate fitness score of the fittest solution at the time of export.
- **Provenance:** These files were generated during the original optimization campaigns and were used during paper preparation. They are provided as-is for cross-checking purposes.
