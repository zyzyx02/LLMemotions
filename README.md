# LLMemotions

This repository contains the full simulation and analysis pipeline used in the study:

**“For those who come after: Emotions of large language models in a post-apocalyptic world”**

The code enables reproduction of the simulation logic, log generation, and statistical analyses reported in the manuscript.

---

## Overview

The project investigates how large language model (LLM)–driven agents exhibit emotionally evolving behavior under long-term narrative pressure. Agents interact, age, and communicate in a simulated post-apocalyptic environment inspired by *Clair Obscur: Expedition 33*, with emotional dynamics analyzed across different temporal resolutions and narrative histories.

---

## Repository Structure


LLMemotions/

├── datasets/

│ └── event_log_1_baseline.csv

│ ├── event_log_1_trauma.csv

│ ├── event_log_12_baseline.csv

│ ├── event_log_12_trauma.csv

│ ├── event_log_52_baseline.csv

│ ├── event_log_52_trauma.csv

│ ├── event_log_356_baseline.csv

│ ├── event_log_356_trauma.csv

│ ├── initial_agents.csv

│ ├── merged_event_logs_dialogue_only.csv

│ ├── sanity_52_baseline_merged.csv

│ └── sanity_52_trauma_merged.cav

├── notebooks/

│ ├── LLMemotions_sentiments.ipynb

│ ├── LLMemotions_sentiments_diffs.ipynb

│ └── LLMemotions_sentiments_SBERT.ipynb

├── src/

│ ├── LLMemotions_full.py

│ └── LLMemotions_multiple_reduced.py

├── CITATION.cff

├── LICENSE

├── README.md

└── requirements.txt


---

## Requirements

- Python 3.10+
- A local installation of `llama-cpp-python`
- A GGUF-format instruction-tuned LLM (e.g., Mistral-7B-Instruct)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Full Simulation

The full-scale simulation used in the paper can be executed with:

```bash
python src/LLMemotions_full.py
```

Executing this file only starts one simulation. Make sure to adjust the parameters beforehand in the code. Also note that this may require significant computational resources and runtime.

## Running the Sanity-Check Simulation

A reduced-scale version of the simulation (52 days/year, fewer agents and years) is provided to enable fast replication:

```bash
python src/LLMemotions_multiple_reduced.py
```

This script generates separate CSV log files for baseline and trauma conditions using fixed random seeds. It is also possible to change the simulation parameters as you like.

## Datasets

The datasets used in the study can be found in the Datasets folder. 

## Statistical Analysis

To reproduce the statistical analyses reported in the manuscript use the files in the notebooks folder which contain Google Colab files.


Run the following to receive TextBlob, VADER, and BERT statistics:

```bash
python notebooks/LLMemotions_sentiments.ipynb
```


Run non-parametric tests and effect size estimation:

```bash
python notebooks/LLMemotions_sentiments_diffs.ipynb
```

The analysis includes:

Mann–Whitney U tests

Kruskal–Wallis tests with Dunn post-hoc comparisons

Benjamini–Hochberg FDR correction

Cliff’s delta effect sizes


Run SBERT:

```bash
python notebooks/LLMemotions_sentiments_SBERT.ipynb
```

## License

This project is released under the MIT License.

## Citation

If you use this code, please cite the associated paper and software release (see CITATION.cff).
