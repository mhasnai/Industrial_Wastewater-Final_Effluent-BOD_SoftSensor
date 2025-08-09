import argparse, os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from bod_soft_sensor.model import build_pipelines, split_X_y

def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    print(f"[{label}] MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f}")
    return mae, rmse, r2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with features + target")
    ap.add_argument("--target", default="F_BOD")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    df = pd.read_csv(args.data)
    predictors = [c for c in df.columns if c != args.target]
    X, y = split_X_y(df, predictors, args.target)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    ext, mlp = build_pipelines(n_features=X.shape[1])

    if args.eval_only:
        # quick fit and report; do not save
        ext.fit(Xtr, ytr)
        yhat = ext.predict(Xte)
        evaluate(yte, yhat, "ExtraTrees (holdout)")
        mlp.fit(Xtr, ytr)
        yhat2 = mlp.predict(Xte)
        evaluate(yte, yhat2, "MLP (holdout)")
        return

    print("Training ExtraTrees...")
    ext.fit(Xtr, ytr)
    yhat = ext.predict(Xte)
    evaluate(yte, yhat, "ExtraTrees (holdout)")
    joblib.dump(ext, os.path.join(args.model_dir, "ext_model.joblib"))

    print("Training MLP...")
    mlp.fit(Xtr, ytr)
    yhat2 = mlp.predict(Xte)
    evaluate(yte, yhat2, "MLP (holdout)")
    joblib.dump(mlp, os.path.join(args.model_dir, "nn_model.joblib"))

if __name__ == "__main__":
    main()