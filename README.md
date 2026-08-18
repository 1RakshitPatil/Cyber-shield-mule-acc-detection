# SHIELD — Mule Account Risk Console

A working, scaled-down implementation of the ML + graph analytics core
described in `SHIELD_Solution_Document_Final.docx`, wrapped in a full
fraud-operations console UI. Built to be demoable in a hackathon setting,
**not** the full production architecture (no real Kafka/Flink/Neo4j/Drools
deployment here — see "Scope" below).

## What's in this build

- **Login** (`Login.tsx`) — session-gated entry point. Demo credentials
  (`analyst` / `shield123` or `admin` / `shield123`), backed by a real
  `/api/auth/login` endpoint on the FastAPI backend. All protected pages
  redirect here if there's no valid session; the case-action endpoint
  (freeze/escalate/dismiss) now requires a valid bearer token and logs the
  real logged-in analyst's name to the audit trail. See the Security
  section below for what's real vs. simplified here.
- **Backend** (`api.py`) — FastAPI service wrapping the trained models:
  paginated/filterable/sortable account queue, per-account SHAP
  explanations, simulated ring-detection endpoint, mock case-action
  endpoint (freeze/escalate/dismiss) with a persistent audit trail.
- **Frontend** (`frontend/`, built from `../shield_frontend`) — a full
  React + TypeScript + Tailwind + React Router console: Overview, Alerts,
  Accounts, Account Detail (with SHAP + simulated ring view + transactions),
  Network & Ring Detection, Investigations, SAR Queue, Model Monitor, Data
  Feeds, Reports, and Settings. Ten pages, one shared design system, one
  user journey: **Login → Overview → Alerts → Accounts → Account Detail →
  Network → Investigations → SAR**.
- **`frontend_classic/`** — the original single-page dashboard from the
  first prototype pass, kept as a lightweight fallback if you ever want it.

## Evaluation criteria mapping

Quick reference for judging — every point below maps to something already
built in this prototype, not a claim about future work.

**Innovation**
- Blends an unsupervised anomaly detector with a supervised classifier
  into one explainable score, rather than a single black-box model or a
  static rule engine.
- Every flagged account carries a live SHAP explanation and plain-language
  narrative, not just a number — a judge can ask "why was this flagged"
  and the system answers, in the UI, in real time.

**Technical Feasibility**
- Not a mockup: `train_model.py` trains real models on the real 9,082-account
  dataset, `api.py` serves real predictions through a documented FastAPI
  service (`/docs` is live, not staged), and the frontend consumes that
  API directly — click Freeze/Escalate/Dismiss in the demo and it's a real
  network call landing in a real audit trail.

**Business Potential**
- Directly targets a named compliance workflow (SAR drafting, investigator
  case management, regulatory feed matching against I4C/CERT-In/NCRP/RBI
  CIBIL) rather than a generic fraud score — the kind of tool a bank's
  fraud-ops team would actually sit in front of daily.
- The audit trail (who acted, when, on what) is the seed of exactly the
  investigator-feedback loop the solution doc's retraining strategy depends
  on — the business value compounds over time, not just at launch.

**Scalability**
- The scoring pipeline already runs against the full 3,924-feature schema,
  not a toy feature set; the API is stateless and horizontally scalable as
  written. `README`'s "Extending this" section below spells out the exact
  next steps (Kafka ingestion, Neo4j graph, Drools rules) without any
  rearchitecting of what's already built.

**User Experience**
- One consistent design system across all ten pages — same sidebar, same
  risk color language, same typography — so an investigator's mental model
  transfers between Alerts, Accounts, and Network instead of relearning a
  new layout each time.
- Dense, sortable, filterable tables built for an analyst doing this all
  day, not a marketing dashboard: consistent risk badges, tabular numeric
  alignment, keyboard-friendly filters.

**Security**
- The console is now login-gated; case-altering actions (freeze/escalate/
  dismiss) require a valid session token, checked server-side, not just
  hidden behind a UI route.
- Every action is attributed to a named, authenticated analyst in the
  audit trail — a bank's compliance team can answer "who froze this
  account and when."
- This is explicitly a **demo-grade** auth layer (see the caveat on the
  login screen and in `api.py`) — SHA-256 hashed demo credentials and an
  in-memory session store, not bcrypt/argon2, SSO, or MFA. That's an
  honest, intentional scope line: it demonstrates the *shape* of an
  auth-gated, audited workflow that a real deployment would harden with
  the bank's actual IAM/SSO stack, not a claim that this is production
  security.

## What's real vs. simulated (read before demoing)

| Area | Status |
|---|---|
| Isolation Forest anomaly score | ✅ Real, trained on the dataset |
| XGBoost classifier | ✅ Real, trained + cross-validated + genuine 20% held-out split |
| Ensemble SHIELD Score (0.25 anomaly + 0.75 XGB, scaled 0–1000) | ✅ Real |
| SHAP explanations | ✅ Real `shap.TreeExplainer` output, per account |
| Accounts / Overview / Account Detail pages | ✅ Backed by the real API and real scores |
| **Validate page / `predict.py`** | ✅ Real — scores any uploaded CSV with the exact persisted training artifacts (scaler, Isolation Forest, categorical encoding, imputation medians, anomaly-score normalization range). Verified to reproduce training-time scores to within floating-point precision (max diff `5.7e-14` across a 500-row test). |
| **No ground-truth labels** | ⚠️ The dataset has no confirmed mule/fraud label. XGBoost is trained on **rule-derived proxy labels**, deliberately built from two narrow, independent signals (occupation/volume mismatch + a bounded extreme-value outlier count) rather than a raw sum of every model feature — see the label-leakage note below. AUC: **0.980 out-of-fold, 0.986 on a genuine 20% held-out split** never touched during fitting. Both numbers are disclosed on-screen (Overview KPI, Model Monitor, disclosure modal) with an explanation of why they're still high. Don't remove either the numbers or the explanation. |
| **Ring Detection / Network graph** | ⚠️ Simulated. The dataset has no account-to-account transaction records, so rings are built by grouping same-occupation, elevated-risk accounts. Clearly labeled "SIMULATED / PROTOTYPE" on screen. |
| **Alerts, Investigations, SAR Queue, Model Monitor trend/drift, Data Feeds** | ⚠️ Simulated. These pages have no backing dataset (no alert log, no transaction ledger, no case-management system, no live feed integration exists). Data is deterministically generated from real account IDs/scores so it's stable across navigation, but it is not real activity — tagged with a "SIMULATED / PROTOTYPE" badge throughout. |
| Kafka ingestion, Neo4j graph DB, Drools rule engine, SAR filing to FIU-IND | ❌ Not built — described in the solution doc as next-phase engineering, not attempted here. |

**Say the caveats out loud to judges.** Transparency about what's real
vs. simulated is a stronger pitch than pretending everything is production
data — and it directly maps to the solution doc's own Section 4.8
feedback-loop design (real investigator decisions → real labels → real
retraining), which this prototype's audit trail is built to feed into.

### A note on label leakage (read this if you're asked about the AUC)

An earlier version of this pipeline built the pseudo-label from a raw sum
of every model input feature, plus the Isolation Forest's own output —
both near-deterministic functions of the same matrix fed to XGBoost. That
produced a ~99.6% AUC that looked impressive but was largely circular.
Cross-validation does **not** catch this kind of leakage, because the
leakage lives in how the label is *defined*, not in how the model is
*fit* — that's why the number stayed high even out-of-fold.

The current version (`train_model.py`) builds the label from two narrow,
documented rule signals instead — an occupation/volume mismatch and a
bounded count of statistically extreme feature values — and deliberately
excludes the Isolation Forest's output from label construction, so the
two ensemble members stay independent. It also carves out a genuine 20%
holdout **before** any fitting or threshold tuning and reports that
number separately from the k-fold OOF figure. Both AUCs are still high
(0.980 / 0.986) — that's expected and disclosed, not a remaining bug: any
proxy label built without confirmed fraud outcomes will correlate with
the features that describe the same accounts to some degree. Treat these
numbers as evidence the pipeline works end-to-end, not as a validated
real-world detection rate.

## Scoring a validation dataset (e.g. from judges)

Two ways to score a new CSV with the trained model — both use the exact
same underlying logic (`predict.py`'s `score_dataframe()`), so results are
identical either way:

**In the app (recommended for a live demo):** log in, go to **Validate**
in the sidebar, drag in a CSV, click Run Validation. You'll get a scored
table, risk distribution, a CSV download, and — importantly — a warning
banner if the uploaded file's columns don't match training's schema
exactly (missing columns get imputed with training's medians; unrecognized
columns are dropped; both are reported, not hidden).

**From the command line:**
```bash
cd shield_prototype
python predict.py --input /path/to/validation.csv --output predictions.csv
```

Neither path retrains or refits anything on the new file — both reuse the
scaler, Isolation Forest, categorical category list, imputation medians,
and anomaly-score normalization range persisted by `train_model.py`, so
scores are directly comparable to what's shown elsewhere in the console.

## How to run it


**Fastest path — one server, everything included:**

```bash
cd shield_prototype
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
// if uvicorn didnt run use -  " python -m uvicorn api:app --reload --port 8000 "
```

Open **http://localhost:8000** — the built frontend is served directly by
the API, so this is the only command you need for a demo. API docs (handy
to show judges) are at **http://localhost:8000/docs**.

> If `uvicorn` isn't recognized on your system, use
> `python -m uvicorn api:app --reload --port 8000` instead — same fix as
> the `pip`/`streamlit` PATH issue from earlier.

**If you want to modify the frontend and see live changes**, run it as two
separate dev servers instead:

```bash
# Terminal 1 — backend
cd shield_prototype
  uvicorn api:app --reload --port 8010

# Terminal 2 — frontend (hot reload)
cd shield_frontend
npm install
npm run dev
```
Then open **http://localhost:5173** — Vite's dev server proxies `/api/*`
calls to the backend on port 8010 (see `vite.config.ts`).

After making changes, rebuild and redeploy into the single-server setup:
```bash
cd shield_frontend
npm run build
rm -rf ../shield_prototype/frontend
cp -r dist ../shield_prototype/frontend
```

## Project structure

```
shield_prototype/
├── data/DataSet.csv          # hackathon dataset (add your own — not bundled, 116MB)
├── data_pipeline.py           # cleaning, sentinel (-1) handling, encoding (train + apply modes)
├── train_model.py             # Isolation Forest + XGBoost + SHAP + ensemble scoring + holdout eval
├── predict.py                 # CLI: score any new CSV with the persisted training artifacts
├── api.py                     # FastAPI backend — scoring, SHAP, ring data, /api/validate, auth, audit trail
├── frontend/                  # BUILT React console (served by api.py) — this is dist/ output
├── frontend_classic/          # original single-page dashboard (kept as fallback)
├── artifacts/                 # generated by train_model.py: models, scaler, categories, medians,
│                               # iso-score range, scores, SHAP values, audit_trail.json
├── requirements.txt
└── README.md

shield_frontend/                # React source — edit here, then `npm run build`
├── src/
│   ├── pages/                  # Overview, Alerts, Accounts, AccountDetail, NetworkPage,
│   │                           # Investigations, SarQueue, Validate, ModelMonitor,
│   │                           # DataFeeds, Reports, Settings, Login
│   ├── layout/                 # Sidebar, Header, AppLayout
│   ├── components/             # RiskBadge, RiskScoreBar, MetricCard, Table primitives, NetworkGraph
│   └── lib/
│       ├── api.ts              # real backend calls (including validateDataset)
│       ├── auth.tsx            # session auth context
│       ├── risk.ts             # tier thresholds (0-400/401-650/651-800/801-1000) + colors
│       └── mock/generators.ts  # deterministic mock data for alerts/investigations/SAR/etc.
```

## API endpoints (see `/docs` for full schema)

| Endpoint | Purpose |
|---|---|
| `GET /api/accounts` | Paginated, filterable, sortable risk queue |
| `GET /api/accounts/{id}` | Full detail + SHAP top factors + narrative |
| `GET /api/accounts/{id}/network` | Simulated ring/cluster graph data |
| `GET /api/metrics` | Portfolio KPIs + both AUC figures + the honest disclaimer |
| `GET /api/filters` | Distinct occupation/account-type/segment values for filter dropdowns |
| `POST /api/accounts/{id}/action` | Freeze/escalate/dismiss (auth required), logged to `artifacts/audit_trail.json` |
| `GET /api/audit-trail` | Investigator action history |
| `POST /api/validate` | Upload a CSV (auth required), score it with the trained model, same logic as `predict.py` |
| `POST /api/auth/login`, `/api/auth/me`, `/api/auth/logout` | Demo session auth (see Security note in the evaluation-criteria section) |

## What to say in the demo

- Open on **Login** — mention it's session-gated and case actions require
  auth server-side, then sign in with the demo analyst account.
- Land on **Overview** — real KPIs, the proxy-label AUC caveat front and
  center, not buried in a tooltip.
- **Alerts → click one → drawer** shows the real SHAP explanation pulled
  live from the model, not canned text.
- **Accounts → click a critical account → Account Detail**: real score,
  real SHAP tornado chart, real Freeze/Escalate/Dismiss actions that hit
  the API and write to the audit trail — click one live during the demo.
- **Network**: real ring data from the backend, and the "SIMULATED /
  PROTOTYPE" badge is visible on purpose — frame it as "here's what this
  looks like once we connect a real transaction graph."
- **Validate**: if judges hand you a CSV, this is the page to use live —
  drag it in, click Run Validation, and the results table/CSV export are
  driven by the exact same trained artifacts as everywhere else in the
  app. If the schema doesn't match exactly, the warning banner says so
  instead of silently producing wrong numbers — point that out, it's a
  deliberate design choice, not a bug you're hoping no one notices.
- **Model Monitor**: show both AUC numbers (k-fold OOF and genuine
  held-out) and be ready to explain, briefly, why they're still high —
  it's a property of weak supervision without confirmed labels, not
  overfitting; the explanation is right there on the page if you want to
  just read it aloud.
- Pop open **`/docs`** for 10 seconds — a clean, auto-generated FastAPI
  Swagger UI signals "this is a real API," which is a stronger signal than
  any dashboard polish.

## Extending this into the full architecture

- Swap the proxy-label generator in `train_model.py` for a real label
  column once SAR outcomes are available — the retraining loop `Settings`
  page's Audit Settings section is designed around this feedback path.
- Replace `_build_network()` in `api.py` with real transaction-pair edges
  and swap NetworkX/d3-force for Neo4j + Louvain/PageRank at scale.
- Replace `lib/mock/generators.ts` calls page-by-page with real endpoints
  as each system (alert engine, case management, SAR filing, live feeds)
  comes online — the service layer in `lib/api.ts` was kept separate from
  `lib/mock/` specifically so this is a one-file swap per page, not a
  rewrite.
