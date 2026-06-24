# LCA Surrogate Model with NSGA-II Multi-Objective Optimization

A machine learning surrogate for Life Cycle Assessment (LCA) of additive and conventional manufacturing processes, combined with NSGA-II multi-objective optimization and SHAP-based explainability.

---

## Overview

Full LCA simulations are computationally expensive and time-consuming. This project builds an **XGBoost surrogate model** trained on synthetically generated LCA data that captures the environmental footprint of three manufacturing processes. The surrogate is then used inside an **NSGA-II genetic algorithm** to find Pareto-optimal process configurations across three competing objectives: environmental footprint, energy consumption, and manufacturing time.

---

## Manufacturing Processes

| Process | Material(s) | LCA Reference Footprint |
|---|---|---|
| Conventional Manufacturing | Aluminum | 14.2 kgCO₂eq/kPt |
| LPBF (Laser Powder Bed Fusion) | Aluminum | 21.5 kgCO₂eq/kPt |
| MEX (Material Extrusion) | PETG-CF, ABS | 8.5 kgCO₂eq/kPt |

---

## Pipeline

```
Data Generation → XGBoost Surrogate → NSGA-II Optimization → Preference Layer → SHAP Explainability
```

### 1. Synthetic Data Generation
- 1,000 samples per process/material combination (4,000 total)
- Features: weight (kg), energy (kWh), build time (h), energy/kg, time/kg
- Target: Environmental Footprint [kgCO₂eq/kPt]
- Operating penalty Π = 0.20·ẑ²_w + 0.50·ẑ²_e + 0.30·ẑ²_t propagates process deviations into 16 LCA impact categories

### 2. Surrogate Model (XGBoost)
- Features: process method, material, weight, energy, time, energy/kg, time/kg
- Categorical encoding via OneHotEncoder in a scikit-learn Pipeline
- Hyperparameters: 300 estimators, max_depth=4, lr=0.05, subsample=0.9

### 3. Validation
- Standard 80/20 train-test split
- **Leave-One-Technology-Out (LOTO)** cross-validation to test generalization to unseen processes
- Surrogate predictions at nominal operating points compared against LCA reference values

### 4. NSGA-II Optimization
- **Objectives** (all minimized):
  - Environmental Footprint (surrogate prediction)
  - Energy consumption (kWh)
  - Build time (h)
- Population: 80 | Generations: 60
- Tournament selection, BLX-α crossover, Gaussian mutation
- Run independently per process/material combination

### 5. Preference Layer
- Weighted normalized scoring over the Pareto front
- Baseline weights: w_footprint = 0.7, w_energy = 0.3, w_time = 0.2
- Full sensitivity analysis over all valid weight combinations (Δ = 0.05 grid)
- Perturbation stability analysis (δ ∈ {0.05, 0.10, 0.15, 0.20})

### 6. SHAP Explainability
- TreeExplainer on the XGBoost model
- Summary plot (beeswarm) and global importance bar chart

---

## Output Figures

| File | Description |
|---|---|
| `fig1_loto_validation.png` | Leave-One-Technology-Out R² and MAE% |
| `fig2_reference_comparison.png` | Surrogate vs LCA reference at nominal point |
| `fig3_sensitivity_analysis.png` | Win frequency and stability across weight scenarios |
| `fig4_decision_boundary.png` | 2D decision boundary map (w_footprint vs w_energy) |
| `fig5a_shap_summary.png` | SHAP beeswarm summary plot |
| `fig5b_shap_bar.png` | SHAP mean absolute contribution (bar chart) |

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn xgboost shap scikit-learn
```

Python ≥ 3.9 recommended.

---

## Usage

```bash
python lca_surrogate.py
```

The script will:
1. Generate the synthetic dataset
2. Train and validate the surrogate
3. Run NSGA-II optimization per technology
4. Perform preference sensitivity analysis
5. Compute SHAP values
6. Save all figures to the working directory

---

## Key Results (example run)

- Surrogate test R² > 0.99 on standard split
- LOTO MAE < 5% across all held-out technologies, confirming generalization
- Ranking preserved: MEX < Conventional < LPBF across all weight scenarios
- Winner stability under weight perturbation: > 90%

---

## Project Structure

```
.
├── lca_surrogate.py       # Main script
├── README.md
└── figures/               # Output plots (generated at runtime)
```

---

## Notes

- All data is synthetically generated based on process specifications and LCA impact factors derived from literature values.
- The surrogate does **not** replace a full LCA study — it is intended for rapid design-space exploration and optimization in early-stage process selection.
- Random seed is fixed (`np.random.seed(42)`) for reproducibility.

---

## License

MIT
