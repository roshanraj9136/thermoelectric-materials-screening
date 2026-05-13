#!/usr/bin/env python3
"""Compact local runner for the pasted thermoelectric Colab workflow."""

import json
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


ROOT = os.path.dirname(os.path.abspath(__file__))
ESTM_PATH = os.path.join(ROOT, "data", "estm.xlsx")
OUT_PATH = os.path.join(ROOT, "results_summary.json")

CATIONS = ["Ge", "Sn", "Pb", "Ag", "Sb", "Bi", "Mn", "Ti", "Zn"]
ANIONS = ["Se", "Te", "S"]
ALL_ELEMENTS = CATIONS + ANIONS
R_GAS = 8.314
ENTROPY_THRESHOLD_CATION = 1.51 * R_GAS
EVAL_TEMPS = np.array([300, 400, 500, 600, 700, 800], dtype=float)


def configurational_entropy(fracs):
    fracs = np.asarray(fracs, dtype=float)
    fracs = fracs[fracs > 1e-12]
    return float(-R_GAS * np.sum(fracs * np.log(fracs)))


def generate_sublattice(n_elements, total_units, step):
    out = []

    def rec(depth, remaining, cur):
        if depth == n_elements - 1:
            out.append(tuple(cur + [remaining * step]))
            return
        for units in range(remaining + 1):
            rec(depth + 1, remaining - units, cur + [units * step])

    rec(0, total_units, [])
    return out


def build_design_space():
    cation_comps = generate_sublattice(len(CATIONS), 10, 0.10)
    anion_comps = generate_sublattice(len(ANIONS), 2, 0.50)
    rows = []
    for cat in cation_comps:
        if configurational_entropy(cat) >= ENTROPY_THRESHOLD_CATION:
            for an in anion_comps:
                rows.append(list(cat) + list(an))
    arr = np.asarray(rows, dtype=float)
    ent = np.array([configurational_entropy(row) for row in arr])
    return arr, ent


def formula_from_array(arr):
    return "".join(
        f"{el}{arr[i]:.4f}" for i, el in enumerate(ALL_ELEMENTS) if arr[i] > 1e-10
    )


def readable_formula(arr):
    return " ".join(
        f"{el}{arr[i]:.1f}" for i, el in enumerate(ALL_ELEMENTS) if arr[i] > 1e-10
    )


def featurize(formulas, batch_size=1000):
    featurizer = ElementProperty.from_preset("magpie")
    chunks = []
    valid = []
    labels = featurizer.feature_labels()
    for start in range(0, len(formulas), batch_size):
        batch = formulas[start : start + batch_size]
        comps = []
        local_valid = []
        for offset, formula in enumerate(batch):
            try:
                comps.append(Composition(formula))
                local_valid.append(start + offset)
            except Exception:
                pass
        if not comps:
            continue
        df = pd.DataFrame({"composition": comps})
        df = featurizer.featurize_dataframe(
            df, col_id="composition", ignore_errors=True, pbar=False
        )
        df = df.drop(columns=["composition"])
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        chunks.append(df.values)
        valid.extend(local_valid)
        print(f"    featurized {min(start + batch_size, len(formulas))}/{len(formulas)}")
    return np.vstack(chunks), labels, valid


def train_oracle():
    df = pd.read_excel(ESTM_PATH)
    df.columns = [c.strip() for c in df.columns]
    clean = df[
        [
            "Formula",
            "temperature(K)",
            "seebeck_coefficient(μV/K)",
            "electrical_conductivity(S/m)",
            "thermal_conductivity(W/mK)",
        ]
    ].copy()
    clean.columns = ["formula", "temperature", "seebeck", "sigma", "kappa"]
    for col in ["temperature", "seebeck", "sigma", "kappa"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.dropna().reset_index(drop=True)

    formulas = clean["formula"].tolist()
    features, labels, valid = featurize(formulas, batch_size=1000)
    clean = clean.iloc[valid].reset_index(drop=True)
    X = np.hstack([features, clean["temperature"].to_numpy().reshape(-1, 1)])

    targets = {
        "abs_S": np.abs(clean["seebeck"].to_numpy()),
        "log_sigma": np.log10(np.clip(clean["sigma"].to_numpy(), 1e-3, None)),
        "log_kappa": np.log10(np.clip(clean["kappa"].to_numpy(), 1e-6, None)),
    }

    cv = {}
    models = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, y in targets.items():
        scores = []
        for train_idx, test_idx in kf.split(X):
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X[train_idx], y[train_idx])
            scores.append(r2_score(y[test_idx], model.predict(X[test_idx])))
        final = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        final.fit(X, y)
        models[name] = final
        cv[name] = float(np.mean(scores))
    return models, cv, len(clean), clean["formula"].nunique()


def evaluate(models, features):
    n = len(features)
    zt = np.zeros((n, len(EVAL_TEMPS)))
    for i, temp in enumerate(EVAL_TEMPS):
        X = np.hstack([features, np.full((n, 1), temp)])
        s = models["abs_S"].predict(X)
        sigma = 10 ** models["log_sigma"].predict(X)
        kappa = 10 ** models["log_kappa"].predict(X)
        zt[:, i] = (s * 1e-6) ** 2 * sigma * temp / np.clip(kappa, 1e-10, None)
    return zt.max(axis=1), zt.mean(axis=1), zt


def efficiency(zavg, th=800.0, tc=300.0):
    if zavg <= 0:
        return 0.0
    carnot = (th - tc) / th
    root = np.sqrt(1 + zavg)
    return float(carnot * (root - 1) / (root + tc / th))


def main():
    t0 = time.time()
    print("Building enhanced design space...")
    design, entropy = build_design_space()
    print(f"  design space: {len(design)} compositions")
    print(f"  entropy range: {entropy.min():.2f} to {entropy.max():.2f} J/(mol K)")

    print("Training ESTM Random Forest oracle...")
    models, cv, n_rows, n_unique = train_oracle()
    print(f"  ESTM rows used: {n_rows}; unique formulas: {n_unique}")
    print(f"  CV R2: {cv}")

    print("Featurizing design space...")
    formulas = [formula_from_array(row) for row in design]
    features, _, valid = featurize(formulas, batch_size=1000)
    if len(valid) != len(design):
        design = design[valid]
        entropy = entropy[valid]

    print("Evaluating all candidates...")
    zt_max, b_avg, zt_profile = evaluate(models, features)
    top = np.argsort(b_avg)[-10:][::-1]

    rows = []
    for rank, idx in enumerate(top, 1):
        rows.append(
            {
                "rank": rank,
                "formula": readable_formula(design[idx]),
                "zT_max": float(zt_max[idx]),
                "B_avg": float(b_avg[idx]),
                "eta_cpm_percent": 100 * efficiency(float(b_avg[idx])),
                "entropy_J_molK": float(entropy[idx]),
                "zT_profile_300_800K": [float(x) for x in zt_profile[idx]],
            }
        )

    summary = {
        "elapsed_minutes": (time.time() - t0) / 60,
        "design_space_size": int(len(design)),
        "estm_rows_used": int(n_rows),
        "estm_unique_formulas": int(n_unique),
        "cv_r2": cv,
        "top_candidates": rows,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTOP 10 CANDIDATES")
    for row in rows:
        print(
            f"#{row['rank']:02d} {row['formula']} | "
            f"zTmax={row['zT_max']:.3f} Bavg={row['B_avg']:.4f} "
            f"eta={row['eta_cpm_percent']:.2f}%"
        )
    print(f"\nSaved {OUT_PATH}")
    print(f"Elapsed: {summary['elapsed_minutes']:.1f} min")


if __name__ == "__main__":
    main()
