# SOMBRERO

**Solar Oberth Maneuver By Realization of EvolutionaRy Optimization**

Evolutionary neurocontroller for optimizing low-thrust Solar Electric Propulsion (SEP) Solar Oberth Maneuver (SOM) trajectories.
This code accompanies the manuscript:

> Maraqten, N., van Lynden, W., Gómez de Olea Ballester, C., & Hein, A. M. (2026).
> *High-temperature photovoltaics for Solar Electric Propulsion Oberth Maneuver:
> ton-class payload feasibility for 200 AU interstellar-precursor missions.*
> Manuscript under review.

**For scientific results and conclusions, please cite the paper (see [Citation](#citation)).**

---

## What This Repository Reproduces

This public release supports **hot-start reconstruction** of the two paper reference trajectories:

| Case | Config | Chromosome | Paper Payload | Paper t_200AU |
|------|--------|------------|---------------|---------------|
| JGA (Jupiter Gravity Assist) | `configs/paper_jga.json` | `configs/best_chromosome_jga.json` | 3,083 kg | 24.97 yr |
| Direct (no JGA) | `configs/paper_nojga.json` | `configs/best_chromosome_nojga.json` | 1,551 kg | 24.34 yr |

**Hot-start** means: running a single trajectory simulation with the provided optimized chromosome (neural network weights), not re-running the full evolutionary optimization from scratch.

### What This Release Does NOT Claim

- Independent rediscovery of the optimum via full optimization (stochastic, computationally expensive)
- Bit-exact reproduction across all platforms and library versions
- Coverage of all historical development versions or intermediate results

Scalar outputs may vary within ~1% of paper-reported values due to integration tolerance sensitivity, chromosome precision (stored at 4 significant figures), and platform/library differences. All key metrics (payload mass, flight time, power levels) are reproduced to within this tolerance.

---

## Installation

### Requirements

- Python >= 3.10
- See `requirements.txt` for dependencies

### Setup

```bash
# Clone the repository
git clone https://github.com/astronadim/sombrero-trajectory-optimizer.git
cd sombrero-trajectory-optimizer

# Create and activate virtual environment
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note on PyTorch:** The default `requirements.txt` installs CPU-only PyTorch. If you have a CUDA GPU and wish to use it, install PyTorch separately following [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Quickstart

### Reproduce Paper Reference Trajectories (Hot-Start)

```bash
# Direct trajectory (no JGA)
python run.py simulate -c configs/paper_nojga.json -chr configs/best_chromosome_nojga.json

# Jupiter Gravity Assist trajectory
python run.py simulate -c configs/paper_jga.json -chr configs/best_chromosome_jga.json
```

Each command runs in under 1 minute on a modern laptop and produces:
- Console output with key scalar metrics (payload mass, flight time, power, fitness score)
- A summary JSON and trajectory plot in the `outputs/` directory

### Run Evolutionary Optimization (Computationally Expensive)

```bash
python run.py optimize -c configs/paper_jga.json
```

> **Warning:** Full optimization campaigns require hours to days of wall-clock time depending on population size, number of generations, and available CPU cores. The paper reference trajectories were discovered over extended optimization campaigns. This mode is provided for completeness; the hot-start `simulate` mode is the intended reproducibility pathway.

---

## Repository Structure

```
SOMBRERO/
├── run.py                          # CLI entry point (simulate / optimize)
├── config_io.py                    # Configuration loading/saving
├── config_schema.py                # Configuration dataclass schema
├── functions_evolutionary.py       # Evolutionary algorithm and neural network
├── functions_trajectory_simulator_solve_ivp.py  # Trajectory dynamics and integration
├── functions_results_processing.py # Plotting and results export
├── utils_repro.py                  # Reproducibility utilities (seeding, summaries)
├── configs/
│   ├── paper_jga.json              # Paper reference config (JGA case)
│   ├── paper_nojga.json            # Paper reference config (direct case)
│   ├── best_chromosome_jga.json    # Optimized chromosome (JGA)
│   └── best_chromosome_nojga.json  # Optimized chromosome (no JGA)
├── data/
│   ├── README.md                   # Artifact documentation
│   └── paper_artifacts/            # Curated Excel artifacts from paper production
├── docs/
│   ├── reproducibility_scope.md    # Detailed reproducibility scope and limitations
│   ├── paper_availability_statements.md  # Draft data/code availability statements
│   └── manuscript.pdf              # Manuscript PDF (for reviewer reference only)
├── legacy/                         # Historical optimization script (reference only)
├── requirements.txt
├── CITATION.cff
├── LICENSE                         # MIT License
└── README.md
```

---

## Data Artifacts

Curated Excel artifacts from the paper production process are stored in `data/paper_artifacts/`. These contain trajectory time-series and scalar summaries exported during the original optimization campaigns. See `data/README.md` for details.

**Primary reproducibility is via code + configs + chromosomes + `simulate` mode.** The Excel files are supporting cross-check artifacts, not the authoritative source of paper results.

---

## Reproducibility Notes

- **Integration tolerances:** The paper configs use `rtol = atol = 1e-5` for the `simulate` command. During optimization, an adaptive tolerance scheme is used (coarser screening, tighter refinement for high-fitness solutions). Scalar outputs are sensitive to tolerance at the ~0.5–1% level.
- **Chromosome precision:** Stored chromosomes use 4 significant figures (scientific notation). Full float64 precision chromosomes from intermediate optimization runs exist in `outputs/` but are not curated for public release.
- **Platform dependence:** Results depend on NumPy/SciPy integration internals and may differ slightly across OS, compiler, or library versions. The key paper claims (ton-class payload, ~25 yr flight time) are robust to these variations.
- **Seed:** The default random seed is 42. This affects optimization (population initialization) but not `simulate` mode with a fixed chromosome.

For a detailed reproducibility scope statement, see [docs/reproducibility_scope.md](docs/reproducibility_scope.md).

---

## Computational Cost

| Mode | Typical Runtime | Purpose |
|------|----------------|---------|
| `simulate` | < 1 min | Hot-start reference trajectory reconstruction |
| `optimize` (5000 gen, pop 16) | Hours–days | Full evolutionary optimization campaign |

---

## Citation

> **For scientific results, cite the manuscript (paper); the code citation is for software reuse and reproduction only.**

### For scientific results (preferred)

If you use results, conclusions, or architectural findings from this work, please cite the paper:

```bibtex
@unpublished{maraqten2026sombrero,
  author = {Maraqten, Nadim and van Lynden, Willem and G{\'o}mez de Olea Ballester, Carlos and Hein, Andreas M.},
  title  = {High-temperature photovoltaics for Solar Electric Propulsion Oberth Maneuver: ton-class payload feasibility for 200~{AU} interstellar-precursor missions},
  year   = {2026},
  note   = {Manuscript under review}
}
```

### For software reuse

If you reuse or extend the SOMBRERO code, please additionally cite the software repository. See `CITATION.cff` for machine-readable citation metadata.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
