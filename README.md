# Active Learning for High-Entropy Thermoelectric Materials

This repository contains a reproducible local workflow for screening high-entropy thermoelectric compositions using the ESTM dataset and a Random Forest surrogate model. The workflow builds a constrained composition design space, featurizes candidate materials with Magpie descriptors, trains property predictors from ESTM measurements, and ranks candidate compositions by predicted thermoelectric performance.

## Project Highlights

- Screened `96,498` high-entropy candidate compositions.
- Trained Random Forest oracle models on `5,205` cleaned ESTM rows covering `880` unique formulas.
- Evaluated predicted zT profiles from `300 K` to `800 K`.
- Used active learning to evaluate only `80` candidates, or about `0.08%` of the design space.
- Identified a top active-learning candidate with predicted peak zT of `1.586`.
- Includes the ESTM data file, local workflow code, and report PDF.

## Repository Contents

| Path | Description |
| --- | --- |
| `Code_Thermoelectrics.py` | Complete thermoelectric screening and active learning code. |
| `run_colab_workflow.py` | Local Python runner for the complete screening workflow. |
| `data/estm.xlsx` | ESTM thermoelectric dataset used by the workflow. |
| `thermoelectric_report.pdf` | Project report PDF. |
| `COLAB_SETUP.md` | Notes for reproducing the original Google Colab environment. |

## Method

1. Generate candidate high-entropy compositions across selected cation and anion sublattices.
2. Filter compositions using configurational entropy constraints.
3. Featurize formulas with Matminer Magpie composition descriptors.
4. Train Random Forest models for absolute Seebeck coefficient, electrical conductivity, and thermal conductivity.
5. Predict zT over a temperature grid from `300 K` to `800 K`.
6. Rank candidates by average predicted zT and estimated CPM efficiency.

## Final Results

The active-learning loop discovered the following top three candidate compositions:

| Rank | Candidate composition | zT max | Mean zT, 300-800 K | CPM efficiency |
| ---: | --- | ---: | ---: | ---: |
| 1 | `Sn0.10 Ag0.20 Sb0.40 Bi0.10 Mn0.10 Zn0.10 Te1.00` | `1.586` | `1.1181` | `15.55%` |
| 2 | `Sn0.20 Ag0.10 Sb0.40 Bi0.10 Mn0.10 Zn0.10 Te1.00` | `1.486` | `1.0575` | `15.00%` |
| 3 | `Sn0.10 Ag0.10 Sb0.40 Bi0.10 Mn0.20 Zn0.10 Te1.00` | `1.442` | `1.0052` | `14.52%` |

Model validation highlights:

- RF surrogate cross-validation R2: `0.853` for absolute Seebeck coefficient, `0.906` for electrical conductivity, and `0.952` for thermal conductivity.
- Active learning sampled `80 / 96,498` candidates.
- `31 / 80` sampled candidates had predicted zT above `1.0`.
- Holdout literature trend validation passed `6 / 6` physics checks.
- SHAP interpretation ranked Sb as the most important element, followed by Ti, Sn, Ge, and Mn.

## Run Locally

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the workflow:

```bash
python run_colab_workflow.py
```

The script reads `data/estm.xlsx` and writes a local JSON summary after completion.

## Notes

The predictions are computational screening results from a surrogate model. Promising candidates should be validated with higher-fidelity calculations and experimental synthesis before drawing materials-performance conclusions.
