"""
SHIELD API — backend for the investigator command console.

Serves the already-trained scores/SHAP values from artifacts/, plus mock
human-in-the-loop actions (freeze / escalate / dismiss) logged to a local
JSON audit trail. Also mounts the static frontend at "/".

Run: uvicorn api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import json
import io
import math
import secrets
import time
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Header, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from predict import load_artifacts, score_dataframe

BASE = Path(__file__).parent
ARTIFACTS = BASE / "artifacts"
AUDIT_LOG = ARTIFACTS / "audit_trail.json"

app = FastAPI(
    title="SHIELD API",
    description=(
        "Mule / fraud-account risk scoring API. Scores blend an unsupervised "
        "Isolation Forest anomaly signal with a supervised XGBoost classifier "
        "trained on rule-derived proxy labels (no confirmed fraud labels exist "
        "in the source dataset — this is disclosed to the end user in the UI, "
        "not hidden)."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load artifacts once at startup
# ---------------------------------------------------------------------------

_scored: pd.DataFrame = pd.read_parquet(ARTIFACTS / "scored_accounts.parquet")
_X: pd.DataFrame = pd.read_parquet(ARTIFACTS / "feature_matrix.parquet")
_shap: np.ndarray = np.load(ARTIFACTS / "shap_values.npy")
_metrics: dict = json.loads((ARTIFACTS / "metrics.json").read_text())
_predict_artifacts = load_artifacts()  # scaler, iso_forest, xgb_model, categories, medians, feature_names

_scored = _scored.set_index("account_id", drop=False)
_id_to_row = {aid: i for i, aid in enumerate(_scored["account_id"].tolist())}

_explanation_cache: dict[int, list[dict]] = {}
_network_cache: dict[int, dict] = {}

if not AUDIT_LOG.exists():
    AUDIT_LOG.write_text("[]")

# ---------------------------------------------------------------------------
# Mock authentication — DEMO ONLY.
#
# This is intentionally lightweight: hardcoded demo accounts, SHA-256 password
# hashing (not bcrypt/argon2), and an in-memory bearer-token session store
# that resets whenever the server restarts. A production deployment would
# replace this with a real identity provider (SSO/OAuth2 against the bank's
# directory), MFA, hashed+salted credentials in a real user store, short-lived
# JWTs with refresh rotation, and RBAC enforced at the API-gateway layer —
# not hand-rolled here. It exists to demonstrate the *shape* of an auth-gated
# workflow (login screen, protected write actions, audit trail tied to a
# named analyst) for the prototype demo, not to be a real security boundary.
# ---------------------------------------------------------------------------
import hashlib


def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


_DEMO_USERS = {
    "analyst": {"password_hash": _hash("shield123"), "display_name": "Demo Analyst", "role": "Analyst"},
    "admin": {"password_hash": _hash("shield123"), "display_name": "Demo Admin", "role": "Compliance Officer"},
}

_SESSIONS: dict[str, dict] = {}  # token -> {username, display_name, role, issued_at}


def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or malformed Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    session = _SESSIONS.get(token)
    if not session:
        raise HTTPException(401, "Invalid or expired session token")
    return session



def _clean(v):
    """Make a numpy/pandas scalar JSON-safe."""
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _risk_tier(score: float) -> str:
    if score >= 801:
        return "critical"
    if score >= 651:
        return "high"
    if score >= 401:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AccountSummary(BaseModel):
    account_id: int
    account_type: str
    occupation: str
    segment: str
    age: Optional[float]
    shield_score: float
    iso_score: float
    xgb_score: float
    flagged: bool
    risk_tier: str


class AccountList(BaseModel):
    total: int
    page: int
    page_size: int
    results: list[AccountSummary]


class ShapContribution(BaseModel):
    feature: str
    value: float
    impact: float
    direction: Literal["increases_risk", "decreases_risk"]


class AccountDetail(AccountSummary):
    branch_code: int
    account_open_date: str
    tenure_bucket: str
    gender: str
    business_type: str
    pseudo_label: int
    top_factors: list[ShapContribution]
    narrative: str


class NetworkNode(BaseModel):
    id: int
    label: str
    risk_tier: str
    shield_score: float
    is_focus: bool


class NetworkEdge(BaseModel):
    source: int
    target: int
    weight: float


class NetworkResponse(BaseModel):
    simulated: bool = True
    note: str
    nodes: list[NetworkNode]
    edges: list[NetworkEdge]


class ActionRequest(BaseModel):
    action: Literal["freeze", "escalate", "dismiss"]
    note: Optional[str] = Field(default="", max_length=500)
    investigator: Optional[str] = "demo_investigator"


class ActionRecord(BaseModel):
    account_id: int
    action: str
    note: str
    investigator: str
    timestamp: float


class MetricsResponse(BaseModel):
    n_accounts: int
    n_features: int
    pseudo_positive_rate: float
    oof_auc: float
    holdout_auc: float
    holdout_size: int
    flagged_count: int
    score_buckets: dict[str, int]
    disclaimer: str


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    username: str
    display_name: str
    role: str


class SessionUser(BaseModel):
    username: str
    display_name: str
    role: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_summary(row: pd.Series) -> dict:
    score = float(row["shield_score"])
    return {
        "account_id": int(row["account_id"]),
        "account_type": row["account_type"],
        "occupation": row["occupation"],
        "segment": row["segment"],
        "age": _clean(row["age"]),
        "shield_score": round(score, 2),
        "iso_score": round(float(row["iso_score"]), 2),
        "xgb_score": round(float(row["xgb_score"]), 4),
        "flagged": bool(row["flagged"]),
        "risk_tier": _risk_tier(score),
    }


def _get_top_factors(account_id: int, k: int = 8) -> list[dict]:
    if account_id in _explanation_cache:
        return _explanation_cache[account_id]
    if account_id not in _id_to_row:
        raise HTTPException(404, f"Account {account_id} not found")
    idx = _id_to_row[account_id]
    row_shap = _shap[idx]
    row_vals = _X.iloc[idx]
    order = np.argsort(-np.abs(row_shap))[:k]
    factors = []
    for i in order:
        impact = float(row_shap[i])
        factors.append({
            "feature": _X.columns[i],
            "value": _clean(row_vals.iloc[i]),
            "impact": round(impact, 4),
            "direction": "increases_risk" if impact > 0 else "decreases_risk",
        })
    _explanation_cache[account_id] = factors
    return factors


def _narrative(account: pd.Series, factors: list[dict]) -> str:
    top = [f for f in factors if f["direction"] == "increases_risk"][:3]
    if not top:
        return "No strong risk-elevating signals detected for this account."
    readable = [f["feature"].replace("_", " ") for f in top]
    lead = f"Flagged primarily due to {', '.join(readable[:-1])}" if len(readable) > 1 else f"Flagged primarily due to {readable[0]}"
    if len(readable) > 1:
        lead += f" and {readable[-1]}"
    tail = f", against an occupation profile of '{account['occupation']}' and account type '{account['account_type']}'."
    return lead + tail


def _build_network(account_id: int) -> dict:
    """Simulated ring — NOT derived from real account-to-account transaction
    links (the source dataset has none). Built by grouping same-tier,
    similar-profile accounts to illustrate what the graph layer would
    surface once real transaction data is available."""
    if account_id in _network_cache:
        return _network_cache[account_id]
    if account_id not in _id_to_row:
        raise HTTPException(404, f"Account {account_id} not found")

    focus = _scored.loc[account_id]
    tier = _risk_tier(focus["shield_score"])
    pool = _scored[
        (_scored["occupation"] == focus["occupation"])
        & (_scored["shield_score"] >= 250)
        & (_scored["account_id"] != account_id)
    ].sort_values("shield_score", ascending=False).head(6)

    nodes = [{
        "id": int(focus["account_id"]),
        "label": f"ACC-{int(focus['account_id'])}",
        "risk_tier": tier,
        "shield_score": round(float(focus["shield_score"]), 1),
        "is_focus": True,
    }]
    edges = []
    rng = np.random.default_rng(int(account_id))
    for _, r in pool.iterrows():
        nodes.append({
            "id": int(r["account_id"]),
            "label": f"ACC-{int(r['account_id'])}",
            "risk_tier": _risk_tier(r["shield_score"]),
            "shield_score": round(float(r["shield_score"]), 1),
            "is_focus": False,
        })
        edges.append({
            "source": int(focus["account_id"]),
            "target": int(r["account_id"]),
            "weight": round(float(rng.uniform(0.3, 0.9)), 2),
        })

    result = {
        "simulated": True,
        "note": (
            "Illustrative only — the source dataset has no account-to-account "
            "transaction records. Edges here group accounts by shared occupation "
            "profile and elevated risk score to demonstrate what the ring-"
            "detection layer would surface once real transaction-graph data "
            "(e.g. from Neo4j in production) is connected."
        ),
        "nodes": nodes,
        "edges": edges,
    }
    _network_cache[account_id] = result
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/auth/login", response_model=LoginResponse)
def login(req: LoginRequest):
    user = _DEMO_USERS.get(req.username.lower())
    if not user or user["password_hash"] != _hash(req.password):
        raise HTTPException(401, "UserName or Password is Invalid")
    token = secrets.token_hex(24)
    _SESSIONS[token] = {
        "username": req.username.lower(),
        "display_name": user["display_name"],
        "role": user["role"],
        "issued_at": time.time(),
    }
    return {"token": token, "username": req.username.lower(), "display_name": user["display_name"], "role": user["role"]}


@app.post("/api/auth/logout")
def logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        _SESSIONS.pop(authorization.removeprefix("Bearer ").strip(), None)
    return {"ok": True}


@app.get("/api/auth/me", response_model=SessionUser)
def me(user: dict = Depends(get_current_user)):
    return {"username": user["username"], "display_name": user["display_name"], "role": user["role"]}


@app.get("/api/accounts", response_model=AccountList)
def list_accounts(
    search: Optional[str] = Query(None, description="Match account_id substring"),
    occupation: Optional[str] = None,
    account_type: Optional[str] = None,
    segment: Optional[str] = None,
    min_score: float = 0,
    max_score: float = 1000,
    flagged_only: bool = False,
    sort_by: str = Query("shield_score", pattern="^(shield_score|iso_score|xgb_score|account_id|age)$"),
    sort_dir: Literal["asc", "desc"] = "desc",
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=10000),
):
    df = _scored
    if search:
        df = df[df["account_id"].astype(str).str.contains(search)]
    if occupation:
        df = df[df["occupation"] == occupation]
    if account_type:
        df = df[df["account_type"] == account_type]
    if segment:
        df = df[df["segment"] == segment]
    df = df[(df["shield_score"] >= min_score) & (df["shield_score"] <= max_score)]
    if flagged_only:
        df = df[df["flagged"] == 1]

    df = df.sort_values(sort_by, ascending=(sort_dir == "asc"))
    total = len(df)
    start = (page - 1) * page_size
    page_df = df.iloc[start:start + page_size]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "results": [_row_to_summary(r) for _, r in page_df.iterrows()],
    }


@app.get("/api/accounts/{account_id}", response_model=AccountDetail)
def get_account(account_id: int):
    if account_id not in _id_to_row:
        raise HTTPException(404, f"Account {account_id} not found")
    row = _scored.loc[account_id]
    factors = _get_top_factors(account_id)
    summary = _row_to_summary(row)
    summary.update({
        "branch_code": int(row["branch_code"]),
        "account_open_date": row["account_open_date"],
        "tenure_bucket": row["tenure_bucket"],
        "gender": row["gender"],
        "business_type": row["business_type"],
        "pseudo_label": int(row["pseudo_label"]),
        "top_factors": factors,
        "narrative": _narrative(row, factors),
    })
    return summary


@app.get("/api/accounts/{account_id}/network", response_model=NetworkResponse)
def get_network(account_id: int):
    return _build_network(account_id)


@app.get("/api/metrics", response_model=MetricsResponse)
def get_metrics():
    scores = _scored["shield_score"]
    buckets = {
        "low (0-400)": int(((scores >= 0) & (scores < 401)).sum()),
        "medium (401-650)": int(((scores >= 401) & (scores < 651)).sum()),
        "high (651-800)": int(((scores >= 651) & (scores < 801)).sum()),
        "critical (801-1000)": int((scores >= 801).sum()),
    }
    return {
        "n_accounts": _metrics["n_accounts"],
        "n_features": _metrics["n_features"],
        "pseudo_positive_rate": _metrics["pseudo_positive_rate"],
        "oof_auc": _metrics["oof_auc"],
        "holdout_auc": _metrics.get("holdout_auc", _metrics["oof_auc"]),
        "holdout_size": _metrics.get("holdout_size", 0),
        "flagged_count": _metrics["flagged_count"],
        "score_buckets": buckets,
        "disclaimer": (
            "This dataset has no confirmed fraud/mule ground-truth labels. "
            "Model was trained on rule-derived proxy labels (occupation/volume "
            "mismatch + a bounded extreme-value outlier count, deliberately NOT "
            "the anomaly detector's own output, to limit label circularity). "
            "Both AUC figures reflect fit to those proxy labels, including the "
            "holdout figure measured on a 20% split never touched during fitting "
            "-- neither is a validated real-world fraud detection rate."
        ),
    }


@app.get("/api/filters")
def get_filter_options():
    return {
        "occupations": sorted(_scored["occupation"].dropna().unique().tolist()),
        "account_types": sorted(_scored["account_type"].dropna().unique().tolist()),
        "segments": sorted(_scored["segment"].dropna().unique().tolist()),
    }


@app.post("/api/accounts/{account_id}/action", response_model=ActionRecord)
def take_action(account_id: int, req: ActionRequest, user: dict = Depends(get_current_user)):
    if account_id not in _id_to_row:
        raise HTTPException(404, f"Account {account_id} not found")
    record = {
        "account_id": account_id,
        "action": req.action,
        "note": req.note or "",
        "investigator": user["display_name"],
        "timestamp": time.time(),
    }
    log = json.loads(AUDIT_LOG.read_text())
    log.append(record)
    AUDIT_LOG.write_text(json.dumps(log, indent=2))
    return record


@app.get("/api/audit-trail", response_model=list[ActionRecord])
def get_audit_trail(account_id: Optional[int] = None):
    log = json.loads(AUDIT_LOG.read_text())
    if account_id is not None:
        log = [r for r in log if r["account_id"] == account_id]
    return sorted(log, key=lambda r: r["timestamp"], reverse=True)


# ---------------------------------------------------------------------------
# Validation dataset upload — score a brand-new CSV with the trained model
# ---------------------------------------------------------------------------

class ValidationRow(BaseModel):
    account_id: int
    iso_score: float
    xgb_score: float
    shield_score: float
    risk_tier: str
    flagged: bool


class ValidationResponse(BaseModel):
    n_rows: int
    flagged_count: int
    mean_score: float
    score_buckets: dict[str, int]
    warnings: list[str]
    results: list[ValidationRow]
    disclaimer: str


@app.post("/api/validate", response_model=ValidationResponse)
async def validate_dataset(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """
    Score an uploaded CSV with the exact artifacts persisted by
    train_model.py -- same categorical encoding, same imputation medians,
    same scaler, same Isolation Forest, same XGBoost model. Nothing is
    refit on the uploaded file, so results are directly comparable to
    scores in the Accounts view. See predict.py for the CLI equivalent.

    Requires auth (this reads an uploaded file and runs inference, which
    is compute work, not a read-only lookup) but does NOT require any
    special role -- any logged-in analyst can validate a file.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file")

    raw = await file.read()
    max_bytes = 250 * 1024 * 1024  # 250MB safety cap for a demo server
    if len(raw) > max_bytes:
        raise HTTPException(413, "File too large for this prototype's demo server (250MB limit)")

    try:
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    except Exception as e:
        raise HTTPException(400, f"Couldn't parse this file as CSV: {e}")

    if df.empty:
        raise HTTPException(400, "Uploaded file has no rows")

    try:
        results, warnings = score_dataframe(df, _predict_artifacts)
    except Exception as e:
        raise HTTPException(422, f"Couldn't score this file: {e}")

    scores = results["shield_score"]
    buckets = {
        "low (0-400)": int(((scores >= 0) & (scores < 401)).sum()),
        "medium (401-650)": int(((scores >= 401) & (scores < 651)).sum()),
        "high (651-800)": int(((scores >= 651) & (scores < 801)).sum()),
        "critical (801-1000)": int((scores >= 801).sum()),
    }

    return {
        "n_rows": len(results),
        "flagged_count": int(results["flagged"].sum()),
        "mean_score": float(scores.mean()),
        "score_buckets": buckets,
        "warnings": warnings,
        "results": [
            {
                "account_id": int(r["account_id"]),
                "iso_score": round(float(r["iso_score"]), 4),
                "xgb_score": round(float(r["xgb_score"]), 4),
                "shield_score": round(float(r["shield_score"]), 2),
                "risk_tier": r["risk_tier"],
                "flagged": bool(r["flagged"]),
            }
            for _, r in results.iterrows()
        ],
        "disclaimer": (
            "Scored using rule-derived proxy labels, not confirmed fraud outcomes "
            "(see /api/metrics disclaimer). Any schema-reconciliation warnings above "
            "indicate columns that didn't match training exactly."
        ),
    }


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

FRONTEND_DIR = BASE / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
