from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from statsmodels.stats.stattools import jarque_bera

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


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, y_true)
    value = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0
    return float(value)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = _mape(y_true, y_pred)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
    }


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_markdown_table(df: pd.DataFrame, path: Path, title: str) -> None:
    headers = list(df.columns)
    rows = [[_format_cell(value) for value in row] for row in df.to_numpy()]
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_error_breakdown_md(
    province_metrics: pd.DataFrame | None,
    size_metrics: pd.DataFrame | None,
    path: Path,
) -> None:
    lines: list[str] = ["# Error Breakdown", ""]
    if province_metrics is not None and not province_metrics.empty:
        lines.append("## By Province")
        lines.append("")
        lines.append("Top 5 by RMSE:")
        lines.append("")
        top = province_metrics.sort_values("rmse", ascending=False).head(5)
        headers = list(top.columns)
        rows = [[_format_cell(value) for value in row] for row in top.to_numpy()]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    if size_metrics is not None and not size_metrics.empty:
        lines.append("## By Hospital Size")
        lines.append("")
        headers = list(size_metrics.columns)
        rows = [[_format_cell(value) for value in row] for row in size_metrics.to_numpy()]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate interpretation reports.")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    settings = Settings()
    paths = get_paths()
    paths.reports.mkdir(parents=True, exist_ok=True)

    features_path = paths.data_processed / "features.parquet"
    model_path = paths.models / "ridge_regression.joblib"
    if not features_path.exists():
        raise FileNotFoundError(f"Features data not found at {features_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    df = pd.read_parquet(features_path)
    df[settings.time_col] = pd.to_datetime(df[settings.time_col])

    split = time_train_test_split(df, settings.time_col, test_size=args.test_size)
    feature_cols = _feature_columns(split.train, settings)
    X_test = split.test[feature_cols]
    y_test = split.test[settings.target_col].to_numpy(dtype=float)

    model = joblib.load(model_path)
    preds = model.predict(X_test)

    # Feature importance (linear model coefficients).
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        importance = pd.DataFrame(
            {
                "feature": feature_cols,
                "coefficient": coef,
                "abs_coefficient": np.abs(coef),
            }
        ).sort_values("abs_coefficient", ascending=False)
        importance_path = paths.reports / "feature_importance.csv"
        importance.to_csv(importance_path, index=False)
        _write_markdown_table(
            importance.head(20),
            paths.reports / "feature_importance.md",
            "Feature Importance (Top 20 by |coef|)",
        )
        logger.info("Wrote feature importance to %s", importance_path)

    # Residual diagnostics.
    residuals = y_test - preds
    jb_stat, jb_pvalue, skew, kurt = jarque_bera(residuals)
    residual_summary = {
        "test_size": float(args.test_size),
        "count": int(len(residuals)),
        "metrics": _compute_metrics(y_test, preds),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_skew": float(skew),
        "residual_kurtosis": float(kurt),
        "jarque_bera": {
            "stat": float(jb_stat),
            "p_value": float(jb_pvalue),
        },
        "residual_quantiles": {
            "p05": float(np.quantile(residuals, 0.05)),
            "p25": float(np.quantile(residuals, 0.25)),
            "p50": float(np.quantile(residuals, 0.50)),
            "p75": float(np.quantile(residuals, 0.75)),
            "p95": float(np.quantile(residuals, 0.95)),
        },
    }
    residual_path = paths.reports / "residual_diagnostics.json"
    residual_path.write_text(json.dumps(residual_summary, indent=2, sort_keys=True))
    residual_md = paths.reports / "residual_diagnostics.md"
    residual_md.write_text(
        "\n".join(
            [
                "# Residual Diagnostics",
                "",
                "## Summary",
                "",
                f"- Test size: {residual_summary['test_size']}",
                f"- Count: {residual_summary['count']}",
                f"- MAE: {residual_summary['metrics']['mae']:.4f}",
                f"- RMSE: {residual_summary['metrics']['rmse']:.4f}",
                f"- MAPE: {residual_summary['metrics']['mape']:.4f}",
                f"- R2: {residual_summary['metrics']['r2']:.4f}",
                "",
                "## Residual Shape",
                "",
                f"- Mean: {residual_summary['residual_mean']:.4f}",
                f"- Std: {residual_summary['residual_std']:.4f}",
                f"- Skew: {residual_summary['residual_skew']:.4f}",
                f"- Kurtosis: {residual_summary['residual_kurtosis']:.4f}",
                f"- Jarque-Bera stat: {residual_summary['jarque_bera']['stat']:.4f}",
                f"- Jarque-Bera p-value: {residual_summary['jarque_bera']['p_value']:.4f}",
                "",
                "## Residual Quantiles",
                "",
                json.dumps(residual_summary["residual_quantiles"], indent=2),
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Error breakdown by province and hospital size.
    breakdown_cols = [settings.hospital_col, settings.target_col]
    if settings.province_col not in breakdown_cols:
        breakdown_cols.append(settings.province_col)
    breakdown_df = split.test[breakdown_cols].copy()
    breakdown_df["prediction"] = preds
    breakdown_df["residual"] = breakdown_df[settings.target_col] - breakdown_df["prediction"]

    province_metrics = None
    if settings.province_col in breakdown_df.columns:
        province_metrics = (
            breakdown_df.groupby(settings.province_col)
            .apply(
                lambda group: pd.Series(
                    _compute_metrics(
                        group[settings.target_col].to_numpy(dtype=float),
                        group["prediction"].to_numpy(dtype=float),
                    )
                    | {"count": int(len(group))}
                )
            )
            .reset_index()
        )
        province_metrics.to_csv(
            paths.reports / "error_breakdown_by_province.csv", index=False
        )

    size_metrics = None
    hospital_means = (
        split.train.groupby(settings.hospital_col)[settings.target_col]
        .mean()
        .rename("avg_target")
    )
    size_bins = pd.qcut(
        hospital_means,
        q=3,
        labels=["small", "medium", "large"],
        duplicates="drop",
    )
    size_map = size_bins.to_dict()
    breakdown_df["hospital_size"] = breakdown_df[settings.hospital_col].map(size_map).fillna(
        "unknown"
    )
    size_metrics = (
        breakdown_df.groupby("hospital_size")
        .apply(
            lambda group: pd.Series(
                _compute_metrics(
                    group[settings.target_col].to_numpy(dtype=float),
                    group["prediction"].to_numpy(dtype=float),
                )
                | {"count": int(len(group))}
            )
        )
        .reset_index()
    )
    size_metrics.to_csv(paths.reports / "error_breakdown_by_size.csv", index=False)

    _write_error_breakdown_md(
        province_metrics, size_metrics, paths.reports / "error_breakdown.md"
    )

    logger.info("Interpretation reports written to %s", paths.reports)
    return 0


if __name__ == "__main__":
    configure_logging()
    raise SystemExit(main())
