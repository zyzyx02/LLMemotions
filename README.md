# LLMemotions

This repository contains the full simulation and analysis pipeline used in the study:

**"For those who come after: How narrative pressure shapes emotional expression in LLM-based agent populations"**

The code enables reproduction of the simulation logic, log generation, and statistical analyses reported in the manuscript.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18250356.svg)](https://doi.org/10.5281/zenodo.18250356)

---

## Overview

The project investigates how large language model (LLM)–driven agents exhibit emotionally evolving behavior under long-term narrative pressure. Agents interact, age, and communicate in a simulated post-apocalyptic environment inspired by *Clair Obscur: Expedition 33*, with emotional dynamics analyzed across different temporal resolutions and narrative histories.

**Note on known limitations:** Approximately 46–49% of LLM dialogue calls return empty output (logged as "---") due to the configured stop sequence; these are excluded from sentiment analyses as described in the manuscript. The mental-state update mechanism uses keyword matching without negation handling; state labels should be treated as indicative annotations. The combat mechanic produces victory only when accumulated training raises agent attack values above the Entity's defense threshold — an emergent property of the simulation, not a scripted outcome. These limitations are discussed in detail in the paper.

---

## Repository Structure

```
LLMemotions/
├── datasets/
│   ├── event_log_1_baseline.csv
│   ├── event_log_1_trauma.csv
│   ├── event_log_12_baseline.csv
│   ├── event_log_12_trauma.csv
│   ├── event_log_52_baseline.csv
│   ├── event_log_52_trauma.csv
│   ├── event_log_365_baseline.csv
│   ├── event_log_365_trauma.csv
│   ├── initial_agents.csv
│   ├── merged_event_logs_dialogue_only.csv
│   ├── sanity_52_baseline_merged.csv
│   └── sanity_52_trauma_merged.csv
├── notebooks/
│   ├── LLMemotions_sentiments.ipynb
│   ├── LLMemotions_sentiments_diffs.ipynb
│   └── LLMemotions_sentiments_SBERT.ipynb
├── reanalysis/
│   ├── reanalysis.py
│   ├── aggregate_full.csv
│   ├── aggregate_valid.csv
│   ├── data_audit.csv
│   ├── direction_consistency.csv
│   ├── state_personality_comparison.csv
│   └── temporal_sentiment.csv
├── src/
│   ├── LLMemotions_full.py
│   └── LLMemotions_multiple_reduced.py
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Requirements

- Python 3.10+
- A local installation of `llama-cpp-python`
- A GGUF-format instruction-tuned LLM (e.g., Mistral-7B-Instruct)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Full Simulation

The full-scale simulation used in the paper can be executed with:

```bash
python src/LLMemotions_full.py
```

Executing this file runs one simulation. Adjust the parameters in the code beforehand. Note that this requires significant computational resources and runtime.

---

## Running the Robustness Simulations

A reduced-scale version of the simulation (52 days/year, 15 agents, 15 years) is provided to enable fast replication:

```bash
python src/LLMemotions_multiple_reduced.py
```

This script generates separate CSV log files for baseline and trauma conditions using fixed random seeds. Simulation parameters can be adjusted in the code.

---

## Datasets

The event log CSV files used in the study are located in the `datasets/` folder. Each file corresponds to one experimental condition (DPY × narrative history). The `merged_event_logs_dialogue_only.csv` file contains pooled dialogue events across all conditions and is used as input for the sentiment analysis notebooks.

The `reanalysis/` folder contains a supplementary analysis script (`reanalysis.py`) and its outputs. This script re-runs sentiment comparisons on the valid dialogue subset (excluding empty LLM outputs and unlabeled rows) and reports direction consistency between the full and valid datasets.

---

## Statistical Analysis

Analyses are implemented as Google Colab notebooks in the `notebooks/` folder.

**Sentiment scoring** (TextBlob, VADER, BERT):
```
notebooks/LLMemotions_sentiments.ipynb
```

**Non-parametric tests and effect sizes:**
```
notebooks/LLMemotions_sentiments_diffs.ipynb
```
Includes: Mann–Whitney U tests, Kruskal–Wallis tests with Dunn post-hoc comparisons, Benjamini–Hochberg FDR correction, Cliff's delta effect sizes.

**Semantic similarity (SBERT):**
```
notebooks/LLMemotions_sentiments_SBERT.ipynb
```

---

## License

This project is released under the MIT License.

---

## Citation

If you use this code or data, please cite the associated paper and software release:

**Paper:** Guzsvinecz, T. (2026). For those who come after: How narrative pressure shapes emotional expression in LLM-based agent populations. *Multimedia Tools and Applications*.
**Paper is under review!**

**Software:** Guzsvinecz, T. (2026). LLMemotions. Zenodo. https://doi.org/10.5281/zenodo.18250356

See also `CITATION.cff` for machine-readable citation metadata.
