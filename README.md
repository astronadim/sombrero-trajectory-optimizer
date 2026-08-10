# SOMBRERO

**Solar Oberth Maneuver By Realisation of EvolutionaRy Optimisation**

Evolutionary neurocontroller for optimizing low-thrust Solar Electric Propulsion (SEP) Solar Oberth Maneuver (SOM) trajectories.
This code accompanies the manuscript:

> Maraqten, N., van Lynden, W., Gómez de Olea Ballester, C., & Hein, A. M. (2026).
> *High-temperature photovoltaics for solar-electric Oberth maneuvers:
> ton-class payload feasibility for interstellar-precursor missions.*
> Manuscript in preparation.

**For scientific results and conclusions, cite the paper (see [Citation](#citation)).**

---

## What This Repository Reproduces

Hot-start reconstruction of the two paper reference trajectories: a single forward simulation with the provided optimized chromosome (neural-network weights), not a re-run of the evolutionary optimization.

| Case | Config | Chromosome | Paper Payload | Paper t_200AU |
|------|--------|------------|---------------|---------------|
| JGA (Jupiter Gravity Assist) | `configs/paper_jga.json` | `configs/best_chromosome_jga.json` | 3,083 kg | 24.97 yr |
| Direct (no JGA) | `configs/paper_nojga.json` | `configs/best_chromosome_nojga.json` | 1,551 kg | 24.34 yr |

Scalar outputs may vary within ~1% of paper values (4-significant-figure chromosomes, platform/library differences). Re-running `simulate` reproduces the direct case to four significant figures; the JGA case yields a payload of about 3,110 kg (+0.9% relative to the paper value; see `data/README.md`). The release does not claim independent rediscovery of the optima via `optimize`, nor bit-exact cross-platform reproduction. Details: [docs/reproducibility_scope.md](docs/reproducibility_scope.md).

---

## Installation

Python >= 3.10.

```bash
git clone https://github.com/astronadim/sombrero-trajectory-optimizer.git
cd sombrero-trajectory-optimizer
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quickstart

```bash
# Direct trajectory (no JGA)
python run.py simulate -c configs/paper_nojga.json -chr configs/best_chromosome_nojga.json

# Jupiter Gravity Assist trajectory
python run.py simulate -c configs/paper_jga.json -chr configs/best_chromosome_jga.json
```

Each command runs in under a minute and writes the scalar metrics to the console and a summary JSON plus trajectory plot to `outputs/`.

Full optimization (`python run.py optimize -c configs/paper_jga.json`) starts from a random (cold) population and takes hours to days depending on population size, generations, and cores. It is provided for completeness; `simulate` is the intended way to reproduce the paper results. To refine from a published solution instead, set `init.init_mode` to `"warm"` in the config; the warm start seeds the population with the provided chromosome.

---

## Repository Structure

```
run.py                          # CLI entry point (simulate / optimize)
config_io.py, config_schema.py  # Configuration I/O and schema
functions_evolutionary.py       # Evolutionary algorithm and neural network
functions_trajectory_simulator_solve_ivp.py  # Trajectory dynamics and integration
functions_results_processing.py # Plotting and results export
utils_repro.py                  # Seeding and run summaries
configs/                        # Paper configs and optimized chromosomes (both cases)
data/paper_artifacts/           # Excel artifacts from the optimization campaigns (see data/README.md)
docs/reproducibility_scope.md   # Reproducibility scope and limitations
```

---

## Reproducibility Notes

- Paper configs use `rtol = atol = 1e-7`; both reference cases also reproduce across `1e-5`–`1e-10`.
- Stored chromosomes carry 4 significant figures; full-precision intermediates are not included.

---

## Citation

For results and conclusions, cite the paper:

```bibtex
@unpublished{maraqten2026sombrero,
  author = {Maraqten, Nadim and van Lynden, Willem and G{\'o}mez de Olea Ballester, Carlos and Hein, Andreas M.},
  title  = {High-temperature photovoltaics for solar-electric {O}berth maneuvers: ton-class payload feasibility for interstellar-precursor missions},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

If you reuse or extend the code, also cite the software:

```bibtex
@software{maraqten2026sombrero_code,
  author  = {Maraqten, Nadim and van Lynden, Willem and G{\'o}mez de Olea Ballester, Carlos and Hein, Andreas M.},
  title   = {SOMBRERO: Solar Oberth Maneuver By Realisation of EvolutionaRy Optimisation},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/astronadim/sombrero-trajectory-optimizer}
}
```

`CITATION.cff` carries the same metadata; GitHub's *Cite this repository* button returns the paper.

---

## License

MIT. See [LICENSE](LICENSE).
