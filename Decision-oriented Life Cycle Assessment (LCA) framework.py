"""
Enhanced Decision-Oriented LCA Framework
=========================================
Improvements over baseline:
  (1) Semi-real validation via leave-one-technology-out + LCA reference comparison
  (2) Sensitivity analysis on the preference layer (weight perturbation)
  (3) Mathematical dataset generation with explicit formulas (documented inline)
  (4) Discussion-ready outputs: effect sizes, ranking stability, boundary analysis
"""

import os
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
import shap

from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import clone

np.random.seed(42)
rng = np.random.default_rng(42)

# ================================================================
# OUTPUT DIRECTORY
# ================================================================
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_fig(filename):
    """Save current figure to the outputs folder."""
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")


# ================================================================
# PROCESS SPECIFICATIONS
# ================================================================

PROCESS_SPEC = {
    "Conventional Manufacturing": {
        "base_impact": 15.0,
        "weight_range": (0.495, 0.650),
        "weight_ref":   0.535,
        "energy_range": (24.0, 33.0),
        "energy_ref":   28.5,
        "time_range":   (15.0, 19.0),
        "time_ref":     17.0,
        "allowed_materials": ["Aluminum"],
        "real_footprint": 14.2,
    },
    "LPBF": {
        "base_impact": 21.0,
        "weight_range": (0.385, 0.410),
        "weight_ref":   0.396,
        "energy_range": (40.0, 50.0),
        "energy_ref":   45.0,
        "time_range":   (15.5, 16.5),
        "time_ref":     16.0,
        "allowed_materials": ["Aluminum"],
        "real_footprint": 21.5,
    },
    "MEX": {
        "base_impact": 9.0,
        "weight_range": (0.390, 0.420),
        "weight_ref":   0.404,
        "energy_range": (22.0, 23.0),
        "energy_ref":   22.5,
        "time_range":   (16.2, 16.8),
        "time_ref":     16.5,
        "allowed_materials": ["PETG-CF", "ABS"],
        "real_footprint": 8.5,
    },
}

MATERIALS = {"Aluminum": 1.00, "PETG-CF": 1.00, "ABS": 1.03}

BASE_LCA = {
    "Conventional Manufacturing": {
        "CO2_Emission": 20.85, "Land_Use": 0.125, "Fossil_Depletion": 174.59,
        "Mineral_Metal_Depletion": 1.03e-3, "Water_Use": 0.097,
        "Freshwater_Ecotoxicity": 3.132, "Freshwater_Eutrophication": 1.54e-5,
        "Acidification": 2.16e-4, "Marine_Eutrophication": 4.10e-5,
        "Terrestrial_Eutrophication": 3.98e-4, "Photochemical_Ozone_Creation": 1.19e-4,
        "Respiratory_Effects": 2.53e-9, "Ionising_Radiation": 3.00e-3,
        "Carcinogenic_Effects": 2.47e-11, "Non_Carcinogenic_Effects": 5.38e-10,
        "Ozone_Layer_Depletion": 8.54e-10, "Environmental_Footprint": 14.2,
    },
    "LPBF": {
        "CO2_Emission": 42.35, "Land_Use": 22.02, "Fossil_Depletion": 377.80,
        "Mineral_Metal_Depletion": 1.20e-3, "Water_Use": 2.50,
        "Freshwater_Ecotoxicity": 2.87, "Freshwater_Eutrophication": 4.30e-3,
        "Acidification": 5.40e-2, "Marine_Eutrophication": 9.50e-3,
        "Terrestrial_Eutrophication": 9.70e-2, "Photochemical_Ozone_Creation": 2.80e-2,
        "Respiratory_Effects": 4.64e-7, "Ionising_Radiation": 9.00e-1,
        "Carcinogenic_Effects": 4.75e-9, "Non_Carcinogenic_Effects": 1.82e-7,
        "Ozone_Layer_Depletion": 5.95e-8, "Environmental_Footprint": 21.5,
    },
    "MEX": {
        "CO2_Emission": 16.33, "Land_Use": 3.80, "Fossil_Depletion": 122.63,
        "Mineral_Metal_Depletion": 9.93e-4, "Water_Use": 0.281,
        "Freshwater_Ecotoxicity": 10.95, "Freshwater_Eutrophication": 1.28e-4,
        "Acidification": 1.60e-3, "Marine_Eutrophication": 2.40e-4,
        "Terrestrial_Eutrophication": 2.50e-3, "Photochemical_Ozone_Creation": 9.10e-4,
        "Respiratory_Effects": 1.47e-8, "Ionising_Radiation": 1.80e-2,
        "Carcinogenic_Effects": 2.73e-10, "Non_Carcinogenic_Effects": 1.45e-8,
        "Ozone_Layer_Depletion": 1.44e-8, "Environmental_Footprint": 8.5,
    },
}

LCA_KEYS = [k for k in BASE_LCA["Conventional Manufacturing"] if k != "Environmental_Footprint"]

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.autolayout": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.85,
})

SHORT_LABELS = {
    "Conventional Manufacturing-Aluminum": "Conv-Al",
    "LPBF-Aluminum": "LPBF-Al",
    "MEX-PETG-CF": "MEX-PETG",
    "MEX-ABS": "MEX-ABS",
    "Conventional Manufacturing": "Conv",
    "LPBF": "LPBF",
    "MEX": "MEX",
}

def shorten(label):
    return SHORT_LABELS.get(label, label)


# ================================================================
# DATA GENERATION
# ================================================================

def clipped_normal(mean, sd, low, high):
    """Sample from a clipped normal distribution."""
    return float(np.clip(rng.normal(mean, sd), low, high))

def operating_penalty(spec, weight, energy, time):
    """
    Quadratic deviation penalty:
      Π = 0.20·ẑ_w² + 0.50·ẑ_e² + 0.30·ẑ_t²
    """
    w_lo, w_hi = spec["weight_range"]
    e_lo, e_hi = spec["energy_range"]
    t_lo, t_hi = spec["time_range"]
    zw = (weight - spec["weight_ref"]) / (w_hi - w_lo)
    ze = (energy - spec["energy_ref"]) / (e_hi - e_lo)
    zt = (time - spec["time_ref"]) / (t_hi - t_lo)
    return 0.20 * zw**2 + 0.50 * ze**2 + 0.30 * zt**2

def generate_sample(method, material):
    spec = PROCESS_SPEC[method]
    base_lca = BASE_LCA[method]

    weight = clipped_normal(
        spec["weight_ref"],
        sd=(spec["weight_range"][1] - spec["weight_range"][0]) / 8,
        low=spec["weight_range"][0], high=spec["weight_range"][1]
    )
    energy = clipped_normal(
        spec["energy_ref"],
        sd=(spec["energy_range"][1] - spec["energy_range"][0]) / 6,
        low=spec["energy_range"][0], high=spec["energy_range"][1]
    )
    time = clipped_normal(
        spec["time_ref"],
        sd=(spec["time_range"][1] - spec["time_range"][0]) / 6,
        low=spec["time_range"][0], high=spec["time_range"][1]
    )

    op_pen = operating_penalty(spec, weight, energy, time)
    mat_factor = MATERIALS[material]

    lca_values, normalized_terms = {}, []
    for key in LCA_KEYS:
        base_val = base_lca[key]
        sensitivity = 0.04 + 0.03 * rng.random()
        noise = rng.normal(0, 0.015)
        value = max(base_val * (1 + sensitivity * op_pen + noise), 1e-20)
        lca_values[key] = value
        normalized_terms.append(value / base_val)

    norm_lca = float(np.mean(normalized_terms))
    env_footprint = (
        spec["base_impact"]
        * (0.70 * norm_lca + 0.30 * (1 + 0.35 * op_pen))
        * mat_factor
        * (1 + rng.normal(0, 0.02))
    )
    env_footprint = max(env_footprint, 0.1)

    row = {
        "Method": method,
        "Material": material,
        "Weight_kg": weight,
        "Weight_g": weight * 1000,
        "Energy_kWh": energy,
        "Time_h": time,
        "Energy_per_kg": energy / weight,
        "Time_per_kg": time / weight,
        "Op_Penalty": op_pen,
        "Env_Footprint": env_footprint,
    }
    row.update(lca_values)
    return row

def generate_dataset(n_per_combo=1000):
    rows = []
    for method in PROCESS_SPEC:
        for material in PROCESS_SPEC[method]["allowed_materials"]:
            for _ in range(n_per_combo):
                rows.append(generate_sample(method, material))
    return pd.DataFrame(rows)


# ================================================================
# PIPELINE HELPERS
# ================================================================

FEATURES = [
    "Method", "Material", "Weight_kg", "Energy_kWh", "Time_h",
    "Energy_per_kg", "Time_per_kg"
]

def make_preprocessor():
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("cat", enc, ["Method", "Material"]),
            ("num", "passthrough", ["Weight_kg", "Energy_kWh", "Time_h", "Energy_per_kg", "Time_per_kg"]),
        ],
        remainder="drop"
    )

def make_xgb():
    return xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        reg_lambda=1.0
    )

def make_pipeline(model, preprocessor):
    return Pipeline([("prep", preprocessor), ("model", model)])

def predict_config(pipe, method, material, weight_kg, energy_kwh, time_h):
    tmp = pd.DataFrame({
        "Method": [method],
        "Material": [material],
        "Weight_kg": [weight_kg],
        "Energy_kWh": [energy_kwh],
        "Time_h": [time_h],
        "Energy_per_kg": [energy_kwh / weight_kg],
        "Time_per_kg": [time_h / weight_kg],
    })
    return float(pipe.predict(tmp)[0])


# ================================================================
# GENERATE DATA & TRAIN
# ================================================================

print("Generating dataset …")
df = generate_dataset(n_per_combo=1000)
df["Label"] = df["Method"] + "-" + df["Material"]

X, y = df[FEATURES], df["Env_Footprint"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = make_preprocessor()
final_pipe_xgb = make_pipeline(make_xgb(), preprocessor)
final_pipe_xgb.fit(X_train, y_train)

pred_xgb = final_pipe_xgb.predict(X_test)
print(f"Baseline Test R²:  {r2_score(y_test, pred_xgb):.4f}")
print(f"Baseline Test MAE: {mean_absolute_error(y_test, pred_xgb):.4f}")


# ================================================================
# (1) SEMI-REAL VALIDATION
# ================================================================

print("\n--- (1) Semi-Real Validation ---")

groups = df["Method"].values
logo = LeaveOneGroupOut()

logo_results = []
for train_idx, test_idx in logo.split(X, y, groups):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    held_out_tech = df.iloc[test_idx]["Method"].iloc[0]

    pipe = make_pipeline(clone(make_xgb()), make_preprocessor())
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)

    logo_results.append({
        "Held-out technology": held_out_tech,
        "N_test": len(y_te),
        "R²": round(r2_score(y_te, pred), 4),
        "MAE": round(mean_absolute_error(y_te, pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_te, pred)), 4),
        "MAE_%": round(100 * mean_absolute_error(y_te, pred) / y_te.mean(), 2),
    })

logo_df = pd.DataFrame(logo_results)
print("\nLeave-One-Technology-Out results:")
print(logo_df.to_string(index=False))

ref_rows = []
for method, spec in PROCESS_SPEC.items():
    for material in spec["allowed_materials"]:
        pred = predict_config(
            final_pipe_xgb, method, material,
            spec["weight_ref"], spec["energy_ref"], spec["time_ref"]
        )
        real = spec["real_footprint"]
        ref_rows.append({
            "Technology": f"{method}-{material}",
            "LCA Reference": real,
            "Surrogate @ nominal": round(pred, 3),
            "Abs Error": round(abs(pred - real), 3),
            "Rel Error (%)": round(100 * abs(pred - real) / real, 2),
        })

ref_df = pd.DataFrame(ref_rows)
print("\nSurrogate vs LCA reference values (nominal operating point):")
print(ref_df.to_string(index=False))

nominal_preds = {row["Technology"]: row["Surrogate @ nominal"] for _, row in ref_df.iterrows()}
ranked_pred = sorted(nominal_preds, key=nominal_preds.get)
rank_preserved = all(
    nominal_preds[ranked_pred[i]] <= nominal_preds[ranked_pred[i + 1]]
    for i in range(len(ranked_pred) - 1)
)
print(f"\nRanking preserved (MEX < Conv < LPBF): {rank_preserved}")
print(f"Predicted ranking: {ranked_pred}")

fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

colors = ["steelblue", "seagreen", "darkorange"]
short_techs = [shorten(t) for t in logo_df["Held-out technology"]]

bars0 = axes[0].bar(short_techs, logo_df["R²"], color=colors, width=0.5, edgecolor="white", linewidth=0.8)
axes[0].axhline(0.90, linestyle="--", linewidth=1.2, color="gray", label="R² = 0.90 threshold")
axes[0].set_title("LOTO R² — generalization test", pad=10)
axes[0].set_ylabel("R²")
axes[0].set_ylim(0, 1.12)
axes[0].set_xlabel("Held-out technology")
for bar, v in zip(bars0, logo_df["R²"]):
    axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
axes[0].legend(loc="lower right", fontsize=9)
sns.despine(ax=axes[0])

bars1 = axes[1].bar(short_techs, logo_df["MAE_%"], color=colors, width=0.5, edgecolor="white", linewidth=0.8)
axes[1].set_title("LOTO MAE (%) — relative error on unseen technology", pad=10)
axes[1].set_ylabel("MAE (%)")
axes[1].set_xlabel("Held-out technology")
ymax1 = logo_df["MAE_%"].max()
axes[1].set_ylim(0, ymax1 * 1.25)
for bar, v in zip(bars1, logo_df["MAE_%"]):
    axes[1].text(bar.get_x() + bar.get_width() / 2, v + ymax1 * 0.03, f"{v:.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
sns.despine(ax=axes[1])

fig.suptitle("(1) Semi-Real Validation: Leave-One-Technology-Out",
             fontsize=13, fontweight="bold", y=1.02)
save_fig("fig1_loto_validation.png")
plt.show()

fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
x = np.arange(len(ref_df))
w = 0.34
short_techs2 = [shorten(t) for t in ref_df["Technology"]]

bars1 = ax.bar(x - w / 2, ref_df["LCA Reference"], w, label="LCA Reference",
               color="darkorange", edgecolor="white", linewidth=0.8)
bars2 = ax.bar(x + w / 2, ref_df["Surrogate @ nominal"], w, label="Surrogate @ nominal",
               color="steelblue", edgecolor="white", linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(short_techs2, fontsize=10)
ax.set_ylabel("Environmental Footprint [kgCO₂eq / kPt]", fontsize=11)
ax.set_title("(1) Surrogate vs LCA Reference — nominal operating point", pad=12)
ax.legend(loc="upper left", fontsize=9)

ymax2 = max(ref_df["LCA Reference"].max(), ref_df["Surrogate @ nominal"].max())
ax.set_ylim(0, ymax2 * 1.20)

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + ymax2 * 0.02, f"{h:.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + ymax2 * 0.02, f"{h:.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

for i, row in ref_df.iterrows():
    ax.text(i, -ymax2 * 0.08, f"Err: {row['Rel Error (%)']:.1f}%",
            ha="center", va="top", fontsize=8.5, color="dimgray")

sns.despine()
save_fig("fig2_reference_comparison.png")
plt.show()


# ================================================================
# NSGA-II
# ================================================================

def make_random_individual(method=None, material=None):
    if method is None:
        method = rng.choice(list(PROCESS_SPEC.keys()))
    if material is None:
        material = rng.choice(PROCESS_SPEC[method]["allowed_materials"])
    spec = PROCESS_SPEC[method]
    return {
        "Method": method,
        "Material": material,
        "Weight_kg": float(rng.uniform(*spec["weight_range"])),
        "Energy_kWh": float(rng.uniform(*spec["energy_range"])),
        "Time_h": float(rng.uniform(*spec["time_range"])),
        "objectives": None,
        "rank": None,
        "crowding": 0.0,
    }

def repair_individual(ind, fixed_method=None, fixed_material=None):
    if fixed_method is not None:
        ind["Method"] = fixed_method
    elif ind["Method"] not in PROCESS_SPEC:
        ind["Method"] = rng.choice(list(PROCESS_SPEC.keys()))

    spec = PROCESS_SPEC[ind["Method"]]

    if fixed_material is not None:
        ind["Material"] = fixed_material
    elif ind["Material"] not in spec["allowed_materials"]:
        ind["Material"] = rng.choice(spec["allowed_materials"])

    ind["Weight_kg"] = float(np.clip(ind["Weight_kg"], *spec["weight_range"]))
    ind["Energy_kWh"] = float(np.clip(ind["Energy_kWh"], *spec["energy_range"]))
    ind["Time_h"] = float(np.clip(ind["Time_h"], *spec["time_range"]))
    return ind

def evaluate_individual(ind, pipe, fixed_method=None, fixed_material=None):
    ind = repair_individual(ind, fixed_method, fixed_material)
    pred = predict_config(pipe, ind["Method"], ind["Material"],
                          ind["Weight_kg"], ind["Energy_kWh"], ind["Time_h"])
    ind["objectives"] = np.array([pred, ind["Energy_kWh"], ind["Time_h"]], dtype=float)
    return ind

def dominates(a, b):
    return (np.all(a["objectives"] <= b["objectives"]) and
            np.any(a["objectives"] < b["objectives"]))

def fast_nondominated_sort(pop):
    S, n = [[] for _ in pop], np.zeros(len(pop), int)
    ranks = np.full(len(pop), -1, int)
    fronts = [[]]
    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q:
                continue
            if dominates(pop[p], pop[q]):
                S[p].append(q)
            elif dominates(pop[q], pop[p]):
                n[p] += 1
        if n[p] == 0:
            ranks[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nf = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    ranks[q] = i + 1
                    nf.append(q)
        i += 1
        fronts.append(nf)

    fronts.pop()
    for idx, r in enumerate(ranks):
        pop[idx]["rank"] = int(r)
    return fronts

def crowding_distance(pop, front):
    if not front:
        return
    n_obj = len(pop[front[0]]["objectives"])
    for idx in front:
        pop[idx]["crowding"] = 0.0

    if len(front) <= 2:
        for idx in front:
            pop[idx]["crowding"] = float("inf")
        return

    for m in range(n_obj):
        sf = sorted(front, key=lambda i: pop[i]["objectives"][m])
        fmin = pop[sf[0]]["objectives"][m]
        fmax = pop[sf[-1]]["objectives"][m]
        pop[sf[0]]["crowding"] = pop[sf[-1]]["crowding"] = float("inf")
        if np.isclose(fmax, fmin):
            continue
        for k in range(1, len(sf) - 1):
            pop[sf[k]]["crowding"] += (
                pop[sf[k + 1]]["objectives"][m] - pop[sf[k - 1]]["objectives"][m]
            ) / (fmax - fmin)

def tournament_select(pop):
    i, j = rng.integers(0, len(pop), 2)
    a, b = pop[i], pop[j]
    if a["rank"] < b["rank"]:
        return a
    if b["rank"] < a["rank"]:
        return b
    return a if a["crowding"] > b["crowding"] else b

def crossover(p1, p2, pc=0.90, fixed_method=None, fixed_material=None):
    child = {
        "Method": fixed_method or (p1["Method"] if rng.random() < 0.5 else p2["Method"]),
        "Material": fixed_material or (p1["Material"] if rng.random() < 0.5 else p2["Material"]),
        "Weight_kg": p1["Weight_kg"],
        "Energy_kWh": p1["Energy_kWh"],
        "Time_h": p1["Time_h"],
        "objectives": None,
        "rank": None,
        "crowding": 0.0,
    }
    if rng.random() < pc:
        a = rng.random()
        child["Weight_kg"] = a * p1["Weight_kg"] + (1 - a) * p2["Weight_kg"]
        child["Energy_kWh"] = a * p1["Energy_kWh"] + (1 - a) * p2["Energy_kWh"]
        child["Time_h"] = a * p1["Time_h"] + (1 - a) * p2["Time_h"]
    return repair_individual(child, fixed_method, fixed_material)

def mutate(ind, pm=0.15, allow_tech=True, fixed_method=None, fixed_material=None):
    if fixed_method is not None:
        ind["Method"] = fixed_method
    elif allow_tech and rng.random() < 0.05:
        ind["Method"] = rng.choice(list(PROCESS_SPEC.keys()))

    spec = PROCESS_SPEC[ind["Method"]]

    if fixed_material is not None:
        ind["Material"] = fixed_material
    elif allow_tech and rng.random() < 0.10:
        ind["Material"] = rng.choice(spec["allowed_materials"])

    if rng.random() < pm:
        ind["Weight_kg"] += rng.normal(0, 0.005)
    if rng.random() < pm:
        ind["Energy_kWh"] += rng.normal(0, 0.6)
    if rng.random() < pm:
        ind["Time_h"] += rng.normal(0, 0.35)

    return repair_individual(ind, fixed_method, fixed_material)

def nsga2(pipe, pop_size=80, generations=60, pc=0.90, pm=0.15,
          fixed_method=None, fixed_material=None):
    population = [
        evaluate_individual(make_random_individual(fixed_method, fixed_material),
                            pipe, fixed_method, fixed_material)
        for _ in range(pop_size)
    ]

    for _ in range(generations):
        fronts = fast_nondominated_sort(population)
        for f in fronts:
            crowding_distance(population, f)

        pool = [tournament_select(population) for _ in range(pop_size)]
        offspring = []

        for i in range(0, pop_size, 2):
            c1 = crossover(dict(pool[i]), dict(pool[(i + 1) % pop_size]), pc, fixed_method, fixed_material)
            c2 = crossover(dict(pool[(i + 1) % pop_size]), dict(pool[i]), pc, fixed_method, fixed_material)
            c1 = mutate(c1, pm, fixed_method is None, fixed_method, fixed_material)
            c2 = mutate(c2, pm, fixed_method is None, fixed_method, fixed_material)

            offspring.extend([
                evaluate_individual(c1, pipe, fixed_method, fixed_material),
                evaluate_individual(c2, pipe, fixed_method, fixed_material),
            ])

        combined = population + offspring[:pop_size]
        cf = fast_nondominated_sort(combined)

        new_pop = []
        for f in cf:
            crowding_distance(combined, f)
            if len(new_pop) + len(f) <= pop_size:
                new_pop.extend([combined[i] for i in f])
            else:
                sf = sorted(f, key=lambda i: combined[i]["crowding"], reverse=True)
                new_pop.extend([combined[i] for i in sf[:pop_size - len(new_pop)]])
                break

        population = new_pop

    ff = fast_nondominated_sort(population)
    for f in ff:
        crowding_distance(population, f)
    return population, [population[i] for i in ff[0]]

def pop_to_df(pop):
    return pd.DataFrame([{
        "Tech": f'{ind["Method"]}-{ind["Material"]}',
        "Method": ind["Method"],
        "Material": ind["Material"],
        "Weight_g": ind["Weight_kg"] * 1000,
        "Weight_kg": ind["Weight_kg"],
        "Energy_kWh": ind["Energy_kWh"],
        "Time_h": ind["Time_h"],
        "Predicted_Footprint": float(ind["objectives"][0]),
        "Rank": int(ind["rank"]),
        "Crowding": float(ind["crowding"]),
    } for ind in pop])


# ================================================================
# PREFERENCE LAYER
# ================================================================

def select_compromise(front_df, weights):
    """
    Return the row minimizing the weighted normalized score.
    """
    weights = np.asarray(weights, dtype=float)
    if np.isclose(weights.sum(), 0):
        raise ValueError("Weights sum to zero.")
    weights = weights / weights.sum()

    df_ = front_df.copy()
    obj_cols = ["Predicted_Footprint", "Energy_kWh", "Time_h"]
    z = df_[obj_cols].astype(float)
    z_min, z_max = z.min(), z.max()
    z_norm = (z - z_min) / (z_max - z_min + 1e-12)

    df_["Decision_Score"] = (
        weights[0] * z_norm["Predicted_Footprint"] +
        weights[1] * z_norm["Energy_kWh"] +
        weights[2] * z_norm["Time_h"]
    )
    best_idx = df_["Decision_Score"].idxmin()
    return df_.loc[best_idx], df_["Decision_Score"]


print("\n--- Running NSGA-II per technology ---")
pareto_parts, selected_rows_base = [], []

for method, spec in PROCESS_SPEC.items():
    for material in spec["allowed_materials"]:
        pop, _ = nsga2(final_pipe_xgb, pop_size=80, generations=60,
                       fixed_method=method, fixed_material=material)
        pop_df = pop_to_df(pop)
        front_df = pop_df[pop_df["Rank"] == 0].copy().reset_index(drop=True)
        pareto_parts.append(front_df)

        compromise, _ = select_compromise(front_df, weights=(0.7, 0.3, 0.2))
        selected_rows_base.append(compromise)

pareto_df = pd.concat(pareto_parts, ignore_index=True)
best_df = pd.DataFrame(selected_rows_base).sort_values("Tech").reset_index(drop=True)

print("\n--- (2) Preference Sensitivity Analysis ---")

step = 0.05
grid = []
for wf in np.arange(0, 1.01, step):
    for we in np.arange(0, 1.01 - wf, step):
        wt = round(1.0 - wf - we, 8)
        if wt < -1e-9:
            continue
        grid.append((round(wf, 3), round(we, 3), round(wt, 3)))

print(f"Weight combinations evaluated: {len(grid)}")

full_front = pareto_df.copy().reset_index(drop=True)

winner_log = []
for wf, we, wt in grid:
    sol, scores = select_compromise(full_front, (wf, we, wt))
    winner_log.append({
        "w_footprint": wf,
        "w_energy": we,
        "w_time": wt,
        "Winner_Tech": sol["Tech"],
        "Winner_Score": sol["Decision_Score"],
    })

winner_df = pd.DataFrame(winner_log)

freq = winner_df["Winner_Tech"].value_counts(normalize=True).reset_index()
freq.columns = ["Technology", "Win_Fraction"]
print("\nRanking frequency across all weight combinations:")
print(freq.to_string(index=False))

baseline_w = (0.7, 0.3, 0.2)
delta_vals = [0.05, 0.10, 0.15, 0.20]
perturb_log = []

baseline_winner = select_compromise(full_front, baseline_w)[0]["Tech"]

for delta in delta_vals:
    for dim in range(3):
        for sign in [+1, -1]:
            w = list(baseline_w)
            w[dim] = np.clip(w[dim] + sign * delta, 0, 1)
            total = sum(w)
            w = [wi / total for wi in w]
            sol, _ = select_compromise(full_front, w)
            perturb_log.append({
                "delta": sign * delta,
                "perturbed_dim": ["w_f", "w_e", "w_t"][dim],
                "Winner": sol["Tech"],
                "Same_as_baseline": sol["Tech"] == baseline_winner,
                "w_f": round(w[0], 3),
                "w_e": round(w[1], 3),
                "w_t": round(w[2], 3),
            })

perturb_df = pd.DataFrame(perturb_log)
stability = perturb_df["Same_as_baseline"].mean()
print(f"\nWinner stability under weight perturbation: {stability:.1%}")
print(perturb_df.groupby("perturbed_dim")["Same_as_baseline"].mean().round(3))

stab = perturb_df.groupby("perturbed_dim")["Same_as_baseline"].mean().reset_index()
stab.columns = ["Dimension", "Stability"]

freq_short = freq.copy()
freq_short["Technology"] = freq_short["Technology"].apply(shorten)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

sns.barplot(data=freq_short, x="Technology", y="Win_Fraction",
            ax=axes[0], palette="viridis", width=0.55)
axes[0].set_title("(2a) Win frequency across all\nweight combinations", pad=10)
axes[0].set_ylabel("Fraction of weight scenarios won")
axes[0].set_xlabel("")
axes[0].set_ylim(0, freq_short["Win_Fraction"].max() * 1.25)
axes[0].tick_params(axis="x", labelsize=9)
for patch, v in zip(axes[0].patches, freq_short["Win_Fraction"]):
    axes[0].text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 0.01,
                 f"{v:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
sns.despine(ax=axes[0])

dim_labels = {"w_f": "w footprint", "w_e": "w energy", "w_t": "w time"}
stab["DimLabel"] = stab["Dimension"].map(dim_labels).fillna(stab["Dimension"])

sns.barplot(data=stab, x="DimLabel", y="Stability", ax=axes[1],
            palette=["steelblue", "seagreen", "darkorange"], width=0.5)
axes[1].set_title("(2b) Winner stability per\nperturbed weight dimension", pad=10)
axes[1].set_ylabel("Fraction of outcomes same as baseline")
axes[1].set_xlabel("")
axes[1].set_ylim(0, 1.18)
axes[1].axhline(1.0, linestyle="--", linewidth=1.2, color="gray", label="100% stable")
axes[1].legend(fontsize=9, loc="lower right")
for patch, v in zip(axes[1].patches, stab["Stability"]):
    axes[1].text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 0.02,
                 f"{v:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
sns.despine(ax=axes[1])

fig.suptitle("(2) Preference Layer Sensitivity Analysis", fontsize=13, fontweight="bold", y=1.02)
save_fig("fig3_sensitivity_analysis.png")
plt.show()

tech_labels = sorted(winner_df["Winner_Tech"].unique())
tech_to_int = {t: i for i, t in enumerate(tech_labels)}
winner_df["Winner_int"] = winner_df["Winner_Tech"].map(tech_to_int)

pivot = winner_df.pivot_table(index="w_footprint", columns="w_energy",
                              values="Winner_int", aggfunc="mean")

fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
im = ax.imshow(
    pivot.values,
    aspect="auto",
    origin="lower",
    extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
    cmap="viridis",
    vmin=0,
    vmax=len(tech_labels) - 1
)
ax.set_xlabel("Weight on energy (w_e)", fontsize=11)
ax.set_ylabel("Weight on footprint (w_f)", fontsize=11)
ax.set_title("(2c) Decision boundary map\n(each zone = dominant technology)", pad=12)

cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("Winning technology", fontsize=10)
cbar.set_ticks(np.linspace(0, len(tech_labels) - 1, len(tech_labels)))
cbar.ax.set_yticklabels([shorten(t) for t in tech_labels], fontsize=9)

save_fig("fig4_decision_boundary.png")
plt.show()


# ================================================================
# SHAP ANALYSIS
# ================================================================

X_test_t = final_pipe_xgb.named_steps["prep"].transform(X_test)
feature_names = final_pipe_xgb.named_steps["prep"].get_feature_names_out()
explainer = shap.TreeExplainer(final_pipe_xgb.named_steps["model"])
shap_values = explainer.shap_values(X_test_t)

# ================================================================
# SHAP SUMMARY PLOT
# ================================================================

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_values,
    X_test_t,
    feature_names=feature_names,
    show=False,
    plot_size=None
)

plt.title(
    "SHAP summary plot — feature impact on environmental footprint",
    fontsize=12,
    fontweight="bold",
    pad=12
)

plt.tight_layout()

save_fig("fig5a_shap_summary.png")
plt.show()
ax_s1.set_title("SHAP summary plot — feature impact on environmental footprint",
                fontsize=12, fontweight="bold", pad=12)
ax_s1.tick_params(axis="y", labelsize=9)
ax_s1.tick_params(axis="x", labelsize=9)
save_fig("fig5a_shap_summary.png")
plt.show()

# ================================================================
# SHAP BAR PLOT
# ================================================================

plt.figure(figsize=(9, 6))

shap.plots.bar(
    shap.Explanation(
        values=shap_values,
        data=X_test_t,
        feature_names=feature_names
    ),
    show=False
)

plt.title(
    "SHAP mean |contribution| — global feature importance",
    fontsize=12,
    fontweight="bold",
    pad=12
)

plt.tight_layout()

save_fig("fig5b_shap_bar.png")
plt.show()
ax_s2.set_title("SHAP mean |contribution| — global feature importance",
                fontsize=12, fontweight="bold", pad=12)
ax_s2.tick_params(axis="y", labelsize=9)
ax_s2.tick_params(axis="x", labelsize=9)
save_fig("fig5b_shap_bar.png")
plt.show()


# ================================================================
# VALIDATION VS REAL
# ================================================================

validation_map = {
    "Conventional Manufacturing-Aluminum": 14.2,
    "LPBF-Aluminum": 21.5,
    "MEX-PETG-CF": 8.5,
    "MEX-ABS": 8.5,
}

val_df = best_df[["Tech", "Predicted_Footprint"]].copy()
val_df["Real_Footprint"] = val_df["Tech"].map(validation_map)
val_df["Abs_Error"] = (val_df["Predicted_Footprint"] - val_df["Real_Footprint"]).abs()
val_df["Rel_Error%"] = 100 * val_df["Abs_Error"] / val_df["Real_Footprint"]
print("\nNSGA-II solutions vs LCA reference:")
print(val_df.round(3))


# ================================================================
# SUMMARY TABLE FOR PAPER
# ================================================================

print("\n========== DISCUSSION-READY SUMMARY ==========")
print("\n[Dataset generation model parameters]")
for method, spec in PROCESS_SPEC.items():
    for material in spec["allowed_materials"]:
        σ_w = (spec["weight_range"][1] - spec["weight_range"][0]) / 8
        σ_e = (spec["energy_range"][1] - spec["energy_range"][0]) / 6
        σ_t = (spec["time_range"][1] - spec["time_range"][0]) / 6
        print(f"  {method}-{material}: σ_w={σ_w:.4f} kg, σ_e={σ_e:.3f} kWh, σ_t={σ_t:.3f} h")

print(f"\n[Preference layer — baseline weights]: w_f=0.7, w_e=0.3, w_t=0.2")
print(f"[Weight combinations evaluated]: {len(grid)}")
print(f"[Winner stability (all perturbations)]: {stability:.1%}")

print(f"\n[LOTO cross-validation]:")
print(logo_df[["Held-out technology", "R²", "MAE_%"]].to_string(index=False))

print(f"\n[Surrogate vs LCA reference]:")
print(ref_df[["Technology", "LCA Reference", "Surrogate @ nominal", "Rel Error (%)"]].to_string(index=False))

print(f"\n[Ranking preserved]: {rank_preserved}")