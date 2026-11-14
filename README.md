# Support vector regression

Predicts tensile strength of steel from its alloy composition with an RBF support-vector regressor.

## What it does

- Uses carbon, manganese, silicon, chromium, nickel, molybdenum, vanadium, nitrogen, niobium, cobalt, tungsten, aluminum, and titanium as features.
- Holds out 25% of the rows with `random_state=42`.
- Searches `C` and `gamma` with 5-fold cross-validation. Epsilon is fixed at 50.
- Reports test MSE and R² for the best estimator.

## Layout

- `src/a4.py` — the grid search and the test metrics
- `data/steel_strength.csv` — alloy composition and strength columns

## Run

From the repository root:

```bash
pip install -r requirements.txt
python src/a4.py
```

## Stack

Python, pandas, scikit-learn, Matplotlib. Written for MEEN 423.
