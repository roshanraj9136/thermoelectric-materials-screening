# Active Learning for High-Entropy Thermoelectric Materials

This repository contains a reproducible local workflow for screening high-entropy thermoelectric compositions using the ESTM dataset and a Random Forest surrogate model. The workflow builds a constrained composition design space, featurizes candidate materials with Magpie descriptors, trains property predictors from ESTM measurements, and ranks candidate compositions by predicted thermoelectric performance.

## Project Highlights

- Screened `96,498` high-entropy candidate compositions.
- Trained Random Forest oracle models on `5,205` cleaned ESTM rows covering `880` unique formulas.
- Evaluated predicted zT profiles from `300 K` to `800 K`.
- Identified top candidate compositions with predicted peak zT above `3.3`.
- Includes the ESTM data file, local workflow code, and report PDF.

## Repository Contents

| Path | Description |
| --- | --- |
| `run_colab_workflow.py` | Local Python runner for the complete screening workflow. |
| `data/estm.xlsx` | ESTM thermoelectric dataset used by the workflow. |
| `thermoelectric_report.pdf` | Project report PDF. |
| `COLAB_SETUP.md` | Notes for reproducing the original Google Colab environment. |
| `source_colab_url.txt` | Original Colab URL reference. |

## Method

1. Generate candidate high-entropy compositions across selected cation and anion sublattices.
2. Filter compositions using configurational entropy constraints.
3. Featurize formulas with Matminer Magpie composition descriptors.
4. Train Random Forest models for absolute Seebeck coefficient, electrical conductivity, and thermal conductivity.
5. Predict zT over a temperature grid from `300 K` to `800 K`.
6. Rank candidates by average predicted zT and estimated CPM efficiency.

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
