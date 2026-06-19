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

## Executive Summary & Final Verdict

The primary objective was to architect a production-ready model to predict daily bike rental demand (`cnt`). Through an iterative development lifecycle, the final deployment recommendation is the **Optimized Gradient Boosting Regressor**.

### Performance Overview

| Model | MAE (bikes) | RMSE (bikes) | MAE % Mean | ME (bikes) | MPE | CV MAE (mean ± std) |
|---------|---------:|---------:|---------:|---------:|---------:|---------:|
| Baseline Random Forest | 818.63 | 1041.70 | 20.7% | -257.38 | -38.1% | 1064.8 ± 491.9 |
| Random Forest (Optimized) | 739.18 | 918.58 | 18.7% | -131.14 | -26.6% | 892.4 ± 239.9 |
| **Gradient Boosting (Optimized)** | **568.96** | **751.39** | **14.4%** | **-133.16** | **-23.4%** | **808.5 ± 222.4** |

---

## Core Architectural Insights

### Superior Average Accuracy

Gradient Boosting minimizes the Mean Absolute Error (MAE) to **568.96 bikes** (14.4% of mean demand), outperforming the optimized Random Forest by more than **170 bikes per day**.

### Mitigation of Volatile Misses

The reduction in Root Mean Squared Error (RMSE) to **751.39** demonstrates that sequential gradient boosting suppresses large forecasting errors during sharp demand fluctuations.

### Systematic Bias Alignment

Both models exhibit a tendency to overpredict demand during December due to a narrow historical baseline. From an operational perspective, Gradient Boosting manages proportional percentage-based distortion more effectively (**−23.4% MPE**), while a mild over-provisioning bias can serve as a safeguard against costly stockouts.

### Generalization Stability

Evaluated using a rolling 5-fold `TimeSeriesSplit`, Gradient Boosting achieves the lowest average validation error (**808.5 bikes**) and reduces variance to **±222.4 bikes**, providing robust performance across seasonal transitions.

---

## Engineering & Diagnostic Highlights

### 1. Fleet Sizing & Service-Level Curve (Part 2)

Assuming an operational threshold of a maximum of **12 rentals per bike per day**, fleet requirements were estimated as:

$$
n = \left\lceil \frac{\text{cnt}}{12} \right\rceil
$$

Two capacity planning scenarios were considered:

- **$n_{\max}$ (100% Coverage):** Fleet size required to satisfy the highest observed demand day.
- **$n_{95}$ (95% Coverage):** Cost-efficient fleet size covering demand on 95% of days, accepting occasional capacity shortfalls to reduce capital expenditure.

### 2. Time-Series Residual Diagnostics (Part 4)

A comprehensive 2×2 residual diagnostic framework (variance, chronological residuals, ACF, and PACF) revealed three structural weaknesses in standard tree-based models:

#### Heteroscedasticity

Funnel-shaped residual patterns indicated increasing prediction uncertainty on low-demand days.

#### Holiday Bias

A severe over-prediction trough (approximately **−2500 bikes**) emerged during Christmas week, reflecting the model's inability to recognize holiday-specific demand behavior.

#### Residual Autocorrelation

Strong lag-1 and lag-2 correlations suggested that temporal dependencies remained unmodeled.

### 3. Feature Optimization Strategy

To address these shortcomings, targeted features were engineered:

#### Dynamic Lag Features

- `cnt_lag_1`
- `cnt_lag_2`

These features successfully removed most residual autocorrelation, flattening both ACF and PACF structures.

#### Calendar Features

- `is_christmas_week`

This feature provided explicit holiday awareness, substantially reducing systematic overestimation during late December.

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

Open `notebook.ipynb` to reproduce all experiments, figures, diagnostics, and final model evaluations.