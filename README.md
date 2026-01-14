# LLMemotions

This repository contains the full simulation and analysis pipeline used in the study:

**“For those who come after: Emotions of large language models in a post-apocalyptic world”**

The code enables reproduction of the simulation logic, log generation, and statistical analyses reported in the manuscript.

---

## Overview

The project investigates how large language model (LLM)–driven agents exhibit emotionally evolving behavior under long-term narrative pressure. Agents interact, age, and communicate in a simulated post-apocalyptic environment inspired by *Clair Obscur: Expedition 33*, with emotional dynamics analyzed across different temporal resolutions and narrative histories.

---

## Repository Structure


AISimulation/

├── README.md

├── LICENSE

├── CITATION.cff

├── requirements.txt

├── src/

│ ├── simulate_full.py

│ ├── simulate_sanity_52.py

│ └── analysis_stats.py

├── notebooks/

│ ├── 01_merge_logs.ipynb

│ └── 02_stats_and_effect_sizes.ipynb

├── data/

│ └── example_logs/

│ ├── event_log_sanity_52_baseline_seed42.csv

│ └── event_log_sanity_52_trauma_seed42.csv

└── outputs/


---

## Requirements

- Python 3.10+
- A local installation of `llama-cpp-python`
- A GGUF-format instruction-tuned LLM (e.g., Mistral-7B-Instruct)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Running the Sanity-Check Simulation (Recommended)

A reduced-scale version of the simulation (52 days/year, fewer agents and years) is provided to enable fast replication:

python src/simulate_sanity_52.py


This script generates separate CSV log files for baseline and trauma conditions using fixed random seeds.


## Full Simulation

The full-scale simulation used in the paper can be executed with:

python src/simulate_full.py

Note that this may require significant computational resources and runtime.

## Statistical Analysis

To reproduce the statistical analyses reported in the manuscript:

Merge event logs:

python notebooks/01_merge_logs.ipynb


Run non-parametric tests and effect size estimation:

python notebooks/02_stats_and_effect_sizes.ipynb


The analysis includes:

Mann–Whitney U tests

Kruskal–Wallis tests with Dunn post-hoc comparisons

Benjamini–Hochberg FDR correction

Cliff’s delta effect sizes

## License

This project is released under the MIT License.

## Citation

If you use this code, please cite the associated paper and software release (see CITATION.cff).
