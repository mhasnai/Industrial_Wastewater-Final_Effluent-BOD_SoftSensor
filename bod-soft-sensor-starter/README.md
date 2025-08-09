# bod-soft-sensor

Soft-sensor code and demo app accompanying the paper:
**Soft Sensing of Biological Oxygen Demand in Industrial Wastewater Using Machine Learning Models** (Hassnain, Lee, Azhar, 2025).

This repository provides a minimal, reviewer-friendly bundle to train and evaluate the Extra Trees (ExT) model for prediction and a simple MLP (NN) model for simulation-style experiments on *sample* data. It also includes a small Streamlit GUI for quick testing.

> **Note:** The industrial dataset is not public. This repo ships with a tiny synthetic sample to verify the pipeline. Replace with your own data following the format described below.

---

## Quickstart

```bash
# 1) Clone and enter
git clone https://github.com/<you>/bod-soft-sensor.git
cd bod-soft-sensor

# 2) Create an environment (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) Install
pip install -r requirements.txt

# 4) Train (uses examples/sample_data.csv)
python scripts/train.py --data examples/sample_data.csv --target F_BOD --model-dir models

# 5) Inference
python scripts/infer.py --model models/ext_model.joblib --data examples/sample_data.csv --out predictions.csv

# 6) (Optional) Run the mini GUI
streamlit run scripts/app.py
```

---

## Data format

CSV with columns (example minimal set):

- `F_COD`, `Flow`, `Flow_Recycle`, `F_SSN`, `F_TKN`, `F_NO3`, `SRT`, `Flow_Aeration`, `DO`, `F_pH`, `Temperature`, `F_BOD`

The synthetic example lives at `examples/sample_data.csv` with a `F_BOD` target. Real deployments can add more features.

---

## Models

- **ExtraTreesRegressor** (`models/ext_model.joblib`): primary predictor (high R² on test/validation in the paper).
- **MLPRegressor** (`models/nn_model.joblib`): used for simulation/extrapolation experiments.

Both are trained with a simple pipeline including scaling and basic hyperparameters (modifiable via CLI flags).

---

## Reproducing a key check

```bash
python scripts/train.py --data examples/sample_data.csv --target F_BOD --model-dir models --eval-only
```

This prints MAE/RMSE/R² on a small holdout split (for sanity, not for scientific claims).

---

## Model weights

This repo saves models under `models/`. For larger/real models, prefer GitHub Releases or LFS. See `scripts/download_model.py` for a stub you can adapt to retrieve weights from an external host.

---

## Citation

Please cite the paper and, if appropriate, the repository (see `CITATION.cff`).

---

## License

MIT. See `LICENSE`.

---

## Tkinter GUI (your Keras model)


You can run your existing Keras/TF model with the provided Tkinter GUI:

```bash
# (inside the repo and your virtualenv)
python scripts/gui_tk.py
```

Then click **Browse Model** and pick your `.keras`/`.h5` file (or `.pkl`/`.joblib` for scikit-learn).

**Important:** GitHub blocks files >100 MB in normal commits. If your Keras model is large, attach it to a **GitHub Release** and download it locally to run the GUI.
