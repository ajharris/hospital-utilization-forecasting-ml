#!/usr/bin/env bash
set -euo pipefail

# create_issues.sh
# Creates the first 10 GitHub issues for the "Hospital Utilization Forecasting Pipeline" repo.
#
# Usage (from repo root in a Codespace):
#   chmod +x scripts/create_issues.sh
#   ./scripts/create_issues.sh
#
# Requirements:
#   - GitHub CLI installed: gh
#   - Authenticated: gh auth status
#   - Repo already created and this script run inside it

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Error: '$1' is not installed or not on PATH."
    exit 1
  }
}

require_cmd gh
require_cmd git

# Ensure we're in a git repo
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "Error: Not inside a git repository. Run this from your project root."
  exit 1
}

# Ensure gh is authenticated
if ! gh auth status >/dev/null 2>&1; then
  echo "Error: GitHub CLI is not authenticated."
  echo "Run: gh auth login"
  exit 1
fi

# Create a label (optional) and ignore if it already exists
create_label() {
  local name="$1"
  local color="$2"
  local desc="$3"
  gh label create "$name" --color "$color" --description "$desc" 2>/dev/null || true
}

create_label "type:setup" "1f6feb" "Project setup, tooling, environment"
create_label "type:data" "2ea043" "Data ingestion, validation, preprocessing"
create_label "type:features" "a371f7" "Feature engineering"
create_label "type:modeling" "fb8500" "Model training and iteration"
create_label "type:evaluation" "d29922" "Validation, metrics, backtesting"
create_label "type:infra" "8b949e" "Docker, CI, reproducibility tooling"
create_label "type:docs" "6e7681" "Documentation, reporting, presentation"

# Helper to create an issue with title/body/labels
create_issue() {
  local title="$1"
  local body="$2"
  local labels_csv="$3"

  echo "Creating issue: $title"
  gh issue create \
    --title "$title" \
    --body "$body" \
    --label "$labels_csv" \
    >/dev/null
}

ISSUE_1_BODY=$'Goal\nCreate a production-style project scaffold and reproducible environment.\n\nTasks\n- Create directories: src/, data/raw/, data/processed/, configs/, reports/, models/, tests/, scripts/\n- Add: requirements.txt (or pyproject.toml), Makefile, .gitignore, Dockerfile (stub), README.md (stub)\n- Add basic lint/format config if desired (ruff/black), but keep minimal\n\nAcceptance Criteria\n- `make install` installs dependencies\n- Repo imports cleanly (no missing modules)\n- Docker builds successfully (even if it runs a stub pipeline)\n\nNotes\n- Prefer typed functions and logging from the start.\n'

ISSUE_2_BODY=$'Goal\nImplement a data ingestion module that pulls hospital utilization data via `publicdata_ca`, validates basic shape, and persists to disk.\n\nTasks\n- Create `src/ingestion.py`\n- Fetch dataset via `publicdata_ca` (config-driven if possible)\n- Save raw dataset to `data/raw/` as Parquet\n- Record dataset metadata (row count, date range, columns) to `reports/dataset_metadata.json`\n- Add `make ingest`\n\nAcceptance Criteria\n- `make ingest` fetches and saves dataset\n- Output stored as Parquet in `data/raw/`\n- Metadata file created under `reports/`\n\nNotes\n- Avoid hardcoding filepaths; use pathlib and a config.\n'

ISSUE_3_BODY=$'Goal\nAdd schema validation and data integrity checks to ensure reliable downstream modeling.\n\nTasks\n- Create `src/validation.py`\n- Implement checks:\n  - Required columns present\n  - Missing values handling strategy (documented)\n  - Date parsing and continuity checks\n  - Hospital/province identifier consistency checks\n- Produce cleaned dataset in `data/processed/` as Parquet\n- Add `make validate` (optional) or run validation as part of `make ingest`\n- Add unit tests for validation edge cases\n\nAcceptance Criteria\n- Validation failures raise clear exceptions\n- Cleaned dataset saved to `data/processed/`\n- Tests cover at least 3 failure modes\n'

ISSUE_4_BODY=$'Goal\nImplement time-series feature engineering with strict prevention of future leakage.\n\nTasks\n- Create `src/features.py`\n- Generate features:\n  - Lag features (t-1, t-3, t-6)\n  - Rolling mean features (3, 6, 12)\n  - Rolling std (optional)\n  - Month-of-year encoding\n  - Growth rates (pct change)\n  - Province-level aggregates (optional)\n- Persist feature set to `data/processed/features.parquet`\n- Add `make features`\n- Add tests to confirm no future leakage\n\nAcceptance Criteria\n- Features are generated deterministically\n- No leakage (features at time t only use data <= t-1)\n- Feature dataset saved to `data/processed/`\n'

ISSUE_5_BODY=$'Goal\nBuild baseline forecasting model and a simple regression model to establish a reproducible training workflow.\n\nTasks\n- Create `src/train.py`\n- Implement:\n  - Baseline: last observation carried forward\n  - Linear regression (or ridge)\n- Time-aware split (train up to cutoff, test after cutoff)\n- Save trained model artifact to `models/`\n- Save metrics to `reports/metrics.json`\n- Add `make train`\n\nAcceptance Criteria\n- Baseline metrics logged\n- Regression metrics logged\n- Training is reproducible (fixed random seed)\n'

ISSUE_6_BODY=$'Goal\nImplement rolling time-series cross-validation (backtesting) to evaluate generalization realistically.\n\nTasks\n- Create `src/backtest.py`\n- Implement:\n  - Expanding window backtesting\n  - Rolling window backtesting (optional)\n- Metrics per fold:\n  - RMSE\n  - MAE\n  - MAPE\n  - R^2\n- Persist fold metrics to `reports/backtest_metrics.json`\n- Add `make backtest`\n\nAcceptance Criteria\n- Backtesting runs end-to-end\n- Fold results aggregated and saved\n- No temporal leakage\n'

ISSUE_7_BODY=$'Goal\nAdd a hierarchical modeling approach (hospital nested in province) and compare against pooled models.\n\nTasks\n- Add `src/hierarchical.py` (or extend train.py)\n- Implement mixed effects model (statsmodels recommended)\n  - Random intercept for hospital\n  - Province grouping\n- Compare pooled regression vs hierarchical\n- Add diagnostics and comparison summary to `reports/hierarchical_comparison.md`\n\nAcceptance Criteria\n- Hierarchical model trains successfully\n- Comparison results documented\n- Reproducible run\n\nNotes\n- If runtime is heavy, limit to a subset for baseline demonstration and document scaling strategy.\n'

ISSUE_8_BODY=$'Goal\nAdd structured experiment tracking so runs are comparable, reproducible, and auditable.\n\nTasks\n- Define a run record schema:\n  - timestamp\n  - dataset fingerprint/hash\n  - model type\n  - params\n  - metrics\n  - git commit sha (optional)\n- Append records to `reports/experiments.jsonl`\n- Ensure train/backtest write run records\n\nAcceptance Criteria\n- Each `make train` writes a new record\n- Each `make backtest` writes a new record\n- Records are machine-readable and stable\n'

ISSUE_9_BODY=$'Goal\nDockerize the pipeline so it runs consistently in any environment.\n\nTasks\n- Finalize Dockerfile\n- Add entrypoint/command(s) to run:\n  - ingest -> features -> train -> backtest\n- Ensure artifacts are written to mounted volume or repo paths\n- Document docker usage in README\n\nAcceptance Criteria\n- `docker build -t hospital-forecast .` succeeds\n- `docker run --rm hospital-forecast` runs pipeline end-to-end\n- Outputs appear under data/, models/, reports/\n'

ISSUE_10_BODY=$'Goal\nFinalize documentation and add model interpretation outputs to make this repo interview-ready.\n\nTasks\n- Expand README with:\n  - problem framing\n  - dataset source\n  - evaluation strategy\n  - how to run\n  - key results\n- Add interpretation artifacts:\n  - feature importance (where applicable)\n  - residual diagnostics summary\n  - error breakdown by province/hospital size\n- Add a short `reports/summary.md` with findings and limitations\n\nAcceptance Criteria\n- README is complete and polished\n- Reports folder contains interpretable results\n- Repo is presentable without additional context\n'

create_issue "Initialize project structure and reproducible environment" "$ISSUE_1_BODY" "type:setup"
create_issue "Build ingestion module using publicdata_ca" "$ISSUE_2_BODY" "type:data"
create_issue "Add schema validation and data integrity checks" "$ISSUE_3_BODY" "type:data"
create_issue "Implement time-series feature engineering module" "$ISSUE_4_BODY" "type:features"
create_issue "Implement baseline and linear regression model" "$ISSUE_5_BODY" "type:modeling"
create_issue "Implement rolling window backtesting framework" "$ISSUE_6_BODY" "type:evaluation"
create_issue "Add hierarchical modeling approach (hospital nested in province)" "$ISSUE_7_BODY" "type:modeling"
create_issue "Add structured experiment tracking" "$ISSUE_8_BODY" "type:evaluation"
create_issue "Containerize full pipeline with Docker" "$ISSUE_9_BODY" "type:infra"
create_issue "Finalize documentation and add model interpretation outputs" "$ISSUE_10_BODY" "type:docs"

echo "Done. Created 10 issues."
echo "Tip: run 'gh issue list' to confirm."
