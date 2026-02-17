from __future__ import annotations

import logging
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.config import Settings, get_paths
from src.logging_config import configure_logging
from src.utils import time_train_test_split

logger = logging.getLogger(__name__)


def _feature_columns(df: pd.DataFrame, settings: Settings) -> list[str]:
    drop_cols = {
        settings.time_col,
        settings.target_col,
        settings.hospital_col,
        settings.province_col,
    }
    return [col for col in df.columns if col not in drop_cols]


def _numeric_feature_columns(df: pd.DataFrame, settings: Settings) -> list[str]:
    feature_cols = _feature_columns(df, settings)
    numeric = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    return numeric


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, y_true)
    value = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0
    return float(value)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mape = _mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
    }


def _subset_if_needed(
    df: pd.DataFrame,
    time_col: str,
    max_rows: int | None,
) -> tuple[pd.DataFrame, str | None]:
    if max_rows is None or len(df) <= max_rows:
        return df, None
    df_sorted = df.sort_values(time_col)
    subset = df_sorted.tail(max_rows).copy()
    note = (
        f"Subset to last {max_rows} rows for runtime. "
        "Scaling strategy: increase max_rows, fit on a larger machine, "
        "or switch to approximate/regularized mixed effects."
    )
    return subset, note


def _fit_mixedlm(
    model: sm.MixedLM,
    methods: list[str],
    maxiter: int = 200,
) -> tuple[sm.regression.mixed_linear_model.MixedLMResults, str]:
    last_error: Exception | None = None
    for method in methods:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            try:
                result = model.fit(reml=False, method=method, maxiter=maxiter)
                return result, method
            except Exception as exc:  # pragma: no cover - defensive
                last_error = exc
                continue
    if last_error:
        raise last_error
    raise RuntimeError("MixedLM fit failed without raising an exception.")


def run_hierarchical(
    settings: Settings | None = None,
    *,
    test_size: float = 0.2,
    max_rows: int | None = 20000,
) -> Path:
    """Train pooled and hierarchical models, write comparison report."""
    settings = settings or Settings()
    np.random.seed(settings.random_seed)
    paths = get_paths()
    paths.reports.mkdir(parents=True, exist_ok=True)

    features_path = paths.data_processed / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Features data not found at {features_path}")

    df = pd.read_parquet(features_path)
    df[settings.time_col] = pd.to_datetime(df[settings.time_col])
    df, subset_note = _subset_if_needed(df, settings.time_col, max_rows=max_rows)

    split = time_train_test_split(df, settings.time_col, test_size=test_size)
    feature_cols = _numeric_feature_columns(split.train, settings)
    if not feature_cols:
        raise ValueError("No numeric feature columns available for modeling.")

    X_train = split.train[feature_cols]
    y_train = split.train[settings.target_col].to_numpy(dtype=float)
    X_test = split.test[feature_cols]
    y_test = split.test[settings.target_col].to_numpy(dtype=float)

    pooled = Ridge(alpha=1.0)
    pooled.fit(X_train, y_train)
    pooled_preds = pooled.predict(X_test)
    pooled_metrics = _compute_metrics(y_test, pooled_preds)

    formula = f"{settings.target_col} ~ " + " + ".join(feature_cols)
    train_df = split.train.copy()
    test_df = split.test.copy()
    groups = train_df[settings.province_col]
    vc_formula = {"hospital": f"0 + C({settings.hospital_col})"}

    mixed = sm.MixedLM.from_formula(
        formula,
        groups=groups,
        vc_formula=vc_formula,
        re_formula="1",
        data=train_df,
    )
    result, optimizer = _fit_mixedlm(mixed, methods=["lbfgs", "powell", "cg"])
    mixed_preds = result.predict(test_df)
    mixed_metrics = _compute_metrics(y_test, mixed_preds)

    report_path = paths.reports / "hierarchical_comparison.md"
    report_lines = [
        "# Hierarchical Model Comparison",
        "",
        "## Data",
        f"- Rows: {len(df)}",
        f"- Train size: {len(split.train)}",
        f"- Test size: {len(split.test)}",
        f"- Features: {', '.join(feature_cols)}",
    ]
    if subset_note:
        report_lines.append(f"- Note: {subset_note}")
    if settings.hospital_col == settings.province_col:
        report_lines.append(
            "- Note: hospital and province columns are identical; "
            "hierarchical effects will overlap."
        )
    report_lines.extend(
        [
            "",
            "## Models",
            "- Pooled regression: Ridge (alpha=1.0)",
            "- Hierarchical: MixedLM with random intercepts for province and hospital (nested)",
            "",
            "## Metrics (Test)",
            "| Model | RMSE | MAE | MAPE | R^2 |",
            "| --- | --- | --- | --- | --- |",
            (
                f"| Pooled | {pooled_metrics['rmse']:.4f} | {pooled_metrics['mae']:.4f} "
                f"| {pooled_metrics['mape']:.4f} | {pooled_metrics['r2']:.4f} |"
            ),
            (
                f"| Hierarchical | {mixed_metrics['rmse']:.4f} | {mixed_metrics['mae']:.4f} "
                f"| {mixed_metrics['mape']:.4f} | {mixed_metrics['r2']:.4f} |"
            ),
            "",
            "## Diagnostics",
            f"- MixedLM converged: {getattr(result, 'converged', None)}",
            f"- MixedLM optimizer: {optimizer}",
            f"- MixedLM AIC: {result.aic:.4f}",
            f"- MixedLM BIC: {result.bic:.4f}",
            "- Random effects: province-level intercept via `groups`, hospital-level intercept via variance components.",
        ]
    )

    report_path.write_text("\n".join(report_lines))
    logger.info("Wrote hierarchical comparison to %s", report_path)
    return report_path


if __name__ == "__main__":
    configure_logging()
    run_hierarchical()
