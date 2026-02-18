# Summary

## Findings

- Ridge regression materially improves over the naive LOCF baseline on the holdout split (RMSE 39.29 vs 62.08).
- Rolling/lag features dominate the linear model signal (rolling means, short lags, and percent change).
- Residuals are right-skewed and heavy-tailed (Jarque-Bera p-value < 0.001), suggesting non-normal errors.
- Error varies by geography and size bucket; larger hospitals show higher absolute error while smaller hospitals have lower RMSE.

## Limitations

- The dataset uses `GEO` for both hospital and province in the current ingest, which limits true hierarchical separation.
- No external covariates (staffing, bed capacity, outbreaks, policy shifts) are included.
- Long time span (1914–2025) likely contains structural breaks not explicitly modeled.
- The primary model is linear; non-linear models may capture additional dynamics.

## Artifacts

- `feature_importance.md`
- `residual_diagnostics.md`
- `error_breakdown.md`
- `hierarchical_comparison.md`
