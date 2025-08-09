import argparse, joblib, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="CSV with features")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--target", default="F_BOD", help="If present in data, it will be ignored for prediction")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if args.target in df.columns:
        df = df.drop(columns=[args.target])

    pipe = joblib.load(args.model)
    preds = pipe.predict(df)
    out = pd.DataFrame({"prediction": preds})
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()