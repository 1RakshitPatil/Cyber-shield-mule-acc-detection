"""
SHIELD Prototype - Data Pipeline
--------------------------------
Loads the hackathon DataSet.csv, separates identity/categorical metadata
from the ~3,900 engineered numeric features, handles the -1 sentinel
value (per the solution doc: -1 means "not applicable / insufficient
history" and must NOT be imputed like a real missing value), and
produces a clean feature matrix ready for modeling.
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/DataSet.csv"

# Columns identified by inspecting the tail of the dataset (Section 3.1.2
# of the solution doc: "final column group includes categorical product,
# occupational, and segment fields")
META_COLS = {
    "account_type": "F3886",     # Savings / Current / MSME Micro / Corp Adv ...
    "branch_code": "F3887",      # high-cardinality numeric code
    "account_open_date": "F3888",
    "tenure_bucket": "F3889",    # G365D, L365D, L180D ... (account age bucket)
    "segment": "F3890",          # R / SU / M / U
    "occupation": "F3891",       # selfemployed / student / salaried / ...
    "gender": "F3892",           # M / F / O
    "business_type": "F3893",    # RETAIL / CORPORATE
    "age": "F3894",
}

CATEGORICAL_COLS = [
    META_COLS["account_type"],
    META_COLS["segment"],
    META_COLS["occupation"],
    META_COLS["gender"],
    META_COLS["business_type"],
    META_COLS["tenure_bucket"],
]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={df.columns[0]: "account_id"})
    return df


def build_feature_matrix(df: pd.DataFrame, categories: dict | None = None, medians: pd.Series | None = None):
    """
    Returns:
        X          -- clean numeric feature matrix (model input)
        meta       -- human-readable columns for the dashboard
        feature_names -- list of column names in X, in order
        fit_info   -- {"categories": {...}, "medians": {...}} actually used
                      (only meaningfully new when categories/medians args
                      were None, i.e. "train mode" -- this is what train.py
                      persists so predict.py can reuse the exact same
                      encoding/imputation later, instead of recomputing it
                      from whatever validation file happens to be passed in)

    IMPORTANT for validation-set consistency:
    Pass the categories/medians saved from training ("apply mode") when
    scoring any dataset other than the original training data. If you
    don't, categorical dummy columns and imputed values will be refit to
    the new file's own distribution -- which silently produces a different
    (and wrong) feature matrix than the one the model was trained on.
    """
    df = df.copy()

    # ---- metadata for display (kept human-readable, not modeled directly) ----
    meta = pd.DataFrame({"account_id": df["account_id"]})
    for label, col in META_COLS.items():
        if col in df.columns:
            meta[label] = df[col]

    # ---- categorical encoding (fixed category list, not refit per file) ----
    cat_df = df[[c for c in CATEGORICAL_COLS if c in df.columns]].copy()
    cat_df = cat_df.fillna("UNKNOWN").astype(str)

    prefixes = list(META_COLS.keys())[:1] + ["segment", "occupation", "gender",
                                               "business_type", "tenure_bucket"]
    fit_categories = {}
    dummy_frames = []
    for col, prefix in zip(cat_df.columns, prefixes):
        if categories is not None and col in categories:
            # apply mode: fix categories to what training saw. Unseen values
            # in this file map to all-zero dummies instead of new columns;
            # categories training saw but this file lacks still appear
            # (as all-zero columns) so the schema matches exactly.
            cats = categories[col]
        else:
            cats = sorted(cat_df[col].unique().tolist())
        fit_categories[col] = cats
        coded = pd.Categorical(cat_df[col], categories=cats)
        dummy_frames.append(pd.get_dummies(coded, prefix=prefix))
    cat_encoded = pd.concat(dummy_frames, axis=1) if dummy_frames else pd.DataFrame(index=df.index)

    # ---- numeric engineered features (everything else, minus id/date/cat) ----
    drop_cols = set(["account_id", META_COLS["account_open_date"],
                      META_COLS["branch_code"]]) | set(CATEGORICAL_COLS)
    numeric_cols = [c for c in df.columns if c not in drop_cols]
    num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # ---- sentinel value (-1) handling ----
    # -1 means "not applicable / insufficient history" (doc Section 3.1.2).
    # We do NOT let it get averaged into real ratios. Instead:
    #   1) capture how often it fires per row -> its own signal
    #   2) mask it to NaN before imputation so it doesn't distort medians
    sentinel_mask = num_df == -1
    sentinel_count = sentinel_mask.sum(axis=1)
    na_count = num_df.isna().sum(axis=1)

    num_df = num_df.mask(sentinel_mask, np.nan)

    if medians is None:
        # train mode: drop fully-empty / constant columns (no signal), then
        # fit medians on this data
        non_empty = num_df.columns[num_df.notna().any()]
        num_df = num_df[non_empty]
        nunique = num_df.nunique(dropna=True)
        num_df = num_df[nunique[nunique > 1].index]
        fit_medians = num_df.median(numeric_only=True)
        num_df = num_df.fillna(fit_medians)
    else:
        # apply mode: reuse training medians exactly, on training's column
        # set -- don't refit anything from this file's own distribution.
        med = pd.Series(medians)
        keep_cols = [c for c in med.index if c in num_df.columns]
        num_df = num_df.reindex(columns=med.index)  # add any missing cols as NaN
        num_df = num_df.fillna(med)
        fit_medians = med

    # Engineered sparsity signals (these ARE predictive per the doc --
    # e.g. card features absent = no card product = different risk profile)
    engineered = pd.DataFrame({
        "sentinel_count": sentinel_count,
        "na_count": na_count,
        "sparsity_ratio": (sentinel_count + na_count) / max(len(numeric_cols), 1),
    })

    X = pd.concat([num_df.reset_index(drop=True),
                   cat_encoded.reset_index(drop=True),
                   engineered.reset_index(drop=True)], axis=1)

    # final safety net: any remaining inf/na -> 0
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    feature_names = list(X.columns)
    fit_info = {"categories": fit_categories, "medians": fit_medians.to_dict()}
    return X, meta.reset_index(drop=True), feature_names, fit_info


def select_top_features(X: pd.DataFrame, engineered_cols, cat_prefixes, top_k: int = 400):
    """
    Prototype-speed feature selection: keep the categorical dummies and
    engineered sparsity signals always, and rank the remaining engineered
    numeric columns (F1...F3885-ish) by variance, keeping the top_k most
    variable ones. This mirrors the doc's own approach for the Isolation
    Forest ("features selected using mutual information ranking; top 200
    features used") -- we use variance ranking here for speed, which is a
    reasonable proxy since near-constant columns carry little signal.
    """
    always_keep = [c for c in X.columns if c in engineered_cols
                   or any(c.startswith(p) for p in cat_prefixes)]
    candidates = [c for c in X.columns if c not in always_keep]
    variances = X[candidates].var().sort_values(ascending=False)
    keep = variances.head(top_k).index.tolist()
    final_cols = always_keep + keep
    return X[final_cols]


if __name__ == "__main__":
    df = load_raw()
    X, meta, feats, fit_info = build_feature_matrix(df)
    print("Raw shape:", df.shape)
    print("Feature matrix shape:", X.shape)
    print("Meta preview:\n", meta.head())
