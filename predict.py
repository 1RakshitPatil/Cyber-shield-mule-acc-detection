"""
SHIELD Prototype - Validation Inference
----------------------------------------
Scores ANY new CSV (e.g. a judges' validation file) using the exact
artifacts persisted by train_model.py -- same categorical encoding, same
imputation medians, same scaler, same Isolation Forest, same XGBoost
model, same anomaly-score normalization range. Nothing here is refit on
the new file.

Usage:
    python predict.py --input path/to/validation.csv --output predictions.csv

Output CSV columns:
    account_id, iso_score, xgb_score, shield_score, risk_tier, flagged

If the validation file's schema doesn't match training's exactly (missing
columns, unseen categories, extra columns), this script does NOT fail --
it aligns to training's schema (missing numeric columns are imputed with
training's medians, missing categorical values map to all-zero dummies,
extra/unrecognized columns are dropped) and prints a warning summary of
what it had to reconcile, so you can judge for yourself whether the
schema mismatch is large enough to worry about.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from data_pipeline import build_feature_matrix

ARTIFACTS = str(Path(__file__).parent / "artifacts")


def load_artifacts():
    with open(f"{ARTIFACTS}/categorical_categories.json") as f:
        categories = json.load(f)
    with open(f"{ARTIFACTS}/imputation_medians.json") as f:
        medians = json.load(f)
    with open(f"{ARTIFACTS}/feature_names.json") as f:
        feature_names = json.load(f)
    with open(f"{ARTIFACTS}/iso_score_range.json") as f:
        iso_range = json.load(f)

    scaler = joblib.load(f"{ARTIFACTS}/scaler.joblib")
    iso_forest = joblib.load(f"{ARTIFACTS}/iso_forest.joblib")

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(f"{ARTIFACTS}/xgb_model.json")

    return {
        "categories": categories,
        "medians": medians,
        "feature_names": feature_names,
        "iso_range": iso_range,
        "scaler": scaler,
        "iso_forest": iso_forest,
        "xgb_model": xgb_model,
    }


def score_dataframe(df: pd.DataFrame, artifacts: dict, threshold: float = 650.0):
    """
    Core scoring logic shared by the CLI (below) and the /api/validate
    backend endpoint -- kept in one place so the web-upload feature can
    never drift out of sync with the command-line tool.

    Returns (results_df, warnings: list[str])
    """
    warnings = []

    if "account_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "account_id"})

    X_raw, meta, feature_names_new, _ = build_feature_matrix(
        df, categories=artifacts["categories"], medians=artifacts["medians"]
    )

    trained_cols = artifacts["feature_names"]
    missing = [c for c in trained_cols if c not in X_raw.columns]
    extra = [c for c in X_raw.columns if c not in trained_cols]
    if missing:
        warnings.append(
            f"{len(missing)} training feature(s) were absent from this file and were "
            f"filled with 0 (e.g. {missing[:5]}{'...' if len(missing) > 5 else ''})."
        )
    if extra:
        warnings.append(
            f"{len(extra)} column(s) in this file don't correspond to any training "
            f"feature and were ignored (e.g. {extra[:5]}{'...' if len(extra) > 5 else ''})."
        )

    # reindex to the EXACT training column set/order -- the real safety net
    X = X_raw.reindex(columns=trained_cols, fill_value=0)

    scaler = artifacts["scaler"]
    iso_forest = artifacts["iso_forest"]
    X_scaled = scaler.transform(X)

    raw_scores = -iso_forest.score_samples(X_scaled)
    lo, hi = artifacts["iso_range"]["min"], artifacts["iso_range"]["max"]
    iso_norm = np.clip((raw_scores - lo) / (hi - lo), 0.0, 1.0)
    out_of_range = int(((raw_scores < lo) | (raw_scores > hi)).sum())
    if out_of_range:
        warnings.append(
            f"{out_of_range} row(s) had anomaly scores outside the range seen during "
            f"training and were clipped to [0, 1] -- these accounts look more extreme "
            f"(or less) than anything in the training data."
        )

    xgb_scores = artifacts["xgb_model"].predict_proba(X)[:, 1]
    shield_score = (0.25 * iso_norm + 0.75 * xgb_scores) * 1000

    def tier(s):
        if s >= 801:
            return "critical"
        if s >= 651:
            return "high"
        if s >= 401:
            return "medium"
        return "low"

    results = meta.copy()
    results["iso_score"] = iso_norm
    results["xgb_score"] = xgb_scores
    results["shield_score"] = shield_score
    results["risk_tier"] = [tier(s) for s in shield_score]
    results["flagged"] = (shield_score >= threshold).astype(int)

    return results, warnings


def main():
    ap = argparse.ArgumentParser(description="Score a validation CSV with the trained SHIELD model.")
    ap.add_argument("--input", required=True, help="Path to the validation CSV")
    ap.add_argument("--output", default="validation_predictions.csv", help="Where to write predictions")
    ap.add_argument("--threshold", type=float, default=650.0, help="SHIELD score flag threshold (0-1000)")
    args = ap.parse_args()

    print(f"Loading artifacts from {ARTIFACTS}/ ...")
    artifacts = load_artifacts()

    print(f"Reading {args.input} ...")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"Loaded {len(df)} rows.")

    results, warnings = score_dataframe(df, artifacts, threshold=args.threshold)

    if warnings:
        print("\nSchema reconciliation warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\nNo schema mismatches -- validation file's columns matched training exactly.")

    results.to_csv(args.output, index=False)
    print(f"\nWrote {len(results)} scored accounts to {args.output}")
    print(f"Flagged: {int(results['flagged'].sum())} / {len(results)} "
          f"({results['flagged'].mean()*100:.2f}%)")
    print(f"\nNOTE: these scores are produced by a model trained on rule-derived "
          f"proxy labels, not confirmed fraud outcomes -- see train_model.py's "
          f"module docstring for the full caveat.")


if __name__ == "__main__":
    main()
