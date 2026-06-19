# Sixt Data Science Lab | Technical Assessment

This repository contains the complete solution for the Sixt Data Scientist candidate technical task. The project leverages the UCI Bike Sharing dataset to build an operationally aligned demand forecasting architecture, moving from exploratory baseline validation to advanced time-series residual diagnostics and model optimization.

---

## Project Structure

```text
├── README.md                 # Executive summary and deployment documentation
├── SIXT_DataScience_Task202402.ipynb            # Core development notebook with code and inline commentary
├── SIXT_DataScience_Task202402.html             # Rendered HTML export of the complete execution
├── requirements.txt          # Python 3 virtual environment dependency freeze
└── data/
    └── day.csv               # Raw daily aggregate bike sharing dataset
```

---

## Executive Summary

The primary objective was to architect a model to predict daily bike rental demand. 

Following the initial assignment requirements, a baseline Random Forest Regressor was developed. A subsequent time-series residual analysis revealed structural opportunities for feature engineering to alleviate systematic biases. To achieve a comprehensive architectural comparison, both an optimized Random Forest and a sequential Gradient Boosting Regressor were trained on this enhanced dataset. 

Based on an accuracy and stability analysis, the final deployment recommendation is the **Gradient Boosting Regressor**.

### Performance Overview

| Model | MAE (bikes) | RMSE (bikes) | MAE % Mean | ME (bikes) | MPE | CV MAE (mean ± std) |
|---------|---------:|---------:|---------:|---------:|---------:|---------:|
| Baseline Random Forest | 818.63 | 1041.70 | 20.7% | -257.38 | -38.1% | 1064.8 ± 491.9 |
| Random Forest (Optimized) | 739.18 | 918.58 | 18.7% | -131.14 | -26.6% | 892.4 ± 239.9 |
| **Gradient Boosting** | **568.96** | **751.39** | **14.4%** | **-133.16** | **-23.4%** | **808.5 ± 222.4** |

---


## Environment Setup & Replication

To reproduce the analysis and verify the reported metrics, create a clean Python environment and install the frozen dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment

# macOS / Linux
source venv/bin/activate

# Windows
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

Open `SIXT_DataScience_Task202402.ipynb` to reproduce all experiments, figures, diagnostics, and final model evaluations.
