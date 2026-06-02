# Active Learning for High-Entropy Thermoelectric Materials

A local Python workflow for screening high-entropy thermoelectric compositions using the ESTM dataset and a Random Forest surrogate model. The pipeline builds a constrained composition design space, featurizes candidate compositions with Magpie descriptors, trains property predictors from ESTM measurements, and uses active learning to rank candidates by predicted average zT (zT_avg) and the resulting thermoelectric conversion efficiency.

## Highlights

- Design space of 96,498 high-entropy candidate compositions.
- Random Forest surrogate trained on 5,205 cleaned ESTM rows covering 880 unique formulas.
- zT predicted on a 300–800 K grid; zT_avg used as the optimization target and as the input to the efficiency formula.
- Active learning evaluates only 80 candidates (~0.08% of the design space).
- Top candidate: predicted zT_avg of 1.0970 and CPM efficiency of 15.36%.

## Repository Contents

| Path | Description |
| --- | --- |
| `Code_Thermoelectrics.py` | Full screening + active learning pipeline. |
| `data/estm.xlsx` | ESTM thermoelectric dataset. |
| `requirements.txt` | Python dependencies. |
| `COLAB_SETUP.md` | Notes for running on Google Colab. |
| `thermoelectric_report.pdf` | Project report. |

## Method

1. Enumerate candidate high-entropy compositions over a 9-element cation and 3-element anion sublattice.
2. Filter by configurational entropy (ΔS_conf ≥ 1.51R on the cation sublattice).
3. Featurize each composition with Matminer Magpie descriptors (132 features).
4. Train Random Forest models on ESTM for absolute Seebeck coefficient, electrical conductivity, and thermal conductivity.
5. Predict zT(T) on the 300–800 K grid and compute zT_avg = mean zT over that range.
6. Run active learning (GPR + entropy-weighted Expected Improvement) with 22 initial seeds and 6 batches.
7. Rank candidates by zT_avg and report CPM conversion efficiency derived from zT_avg.

## Why zT_avg

The CPM efficiency formula

```
η_CPM = (T_h − T_c) / T_h × (√(1 + zT_avg) − 1) / (√(1 + zT_avg) + T_c / T_h)
```

depends on the average zT across the working temperature range, not the peak. Optimizing zT_max can favour narrow-band performers that look strong at a single temperature but give a lower integrated efficiency over the device operating window. The pipeline therefore uses zT_avg both as the active-learning target and as the input to the efficiency calculation; zT_max is reported only as a reference.

## Final Results

Top three candidates discovered by the active-learning loop:

| Rank | Composition | zT_avg (300–800 K) | η_CPM | zT_max (ref) |
| ---: | --- | ---: | ---: | ---: |
| 1 | `Sn0.10 Ag0.20 Sb0.40 Bi0.10 Mn0.10 Zn0.10 Te1.00` | 1.0970 | 15.36% | 1.542 |
| 2 | `Sn0.20 Ag0.10 Sb0.30 Bi0.10 Mn0.20 Zn0.10 Te1.00` | 1.0548 | 14.98% | 1.506 |
| 3 | `Sn0.20 Ag0.10 Sb0.40 Bi0.10 Mn0.10 Zn0.10 Te1.00` | 1.0461 | 14.90% | 1.469 |

Oracle quality (5-fold CV R²):

- Absolute Seebeck coefficient: 0.853
- log10(electrical conductivity): 0.906
- log10(thermal conductivity): 0.952

Active-learning summary:

- 80 / 96,498 candidates sampled (0.08%).
- 29 / 80 sampled candidates have predicted zT_avg above 1.0.
- Best zT_avg found = 38.3% of the oracle's global maximum.
- Bi-avoidance check passes (AL drives Bi fraction down vs. initial seeds).

Physics validation against 10 literature compositions (zT_avg trend tests within the GeTe family): **6 / 6 passed**.

SHAP element importance (top five): Sb > Ti > Sn > Mn > Ge.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python Code_Thermoelectrics.py
```

End-to-end runtime is ~2 minutes on a modern CPU. The script writes four diagnostic figures (`figure1_al_main.png`, `figure2_transport.png`, `figure3_efficiency.png`, `figure4_holdout_test.png`) into the project directory.

## Notes

These are computational screening results from a surrogate oracle trained on ESTM. The dataset contains no compositions with all 12 elements of the design space, so absolute zT values for high-entropy candidates are extrapolations and should be treated as ranked priorities rather than predictions of measured performance. Promising candidates should be validated with higher-fidelity calculations and experimental synthesis.
