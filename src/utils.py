from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Tuple

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def time_train_test_split(
    df: pd.DataFrame,
    time_col: str,
    test_size: float,
) -> TimeSplit:
    """Split data into time-ordered train/test partitions.

    Args:
        df: Input data.
        time_col: Timestamp column name.
        test_size: Fraction of rows assigned to test set.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    split_idx = int((1.0 - test_size) * len(df_sorted))
    if split_idx <= 0 or split_idx >= len(df_sorted):
        raise ValueError("test_size results in empty train or test split.")

    train = df_sorted.iloc[:split_idx].copy()
    test = df_sorted.iloc[split_idx:].copy()
    return TimeSplit(train=train, test=test)


def rolling_window_cv(
    df: pd.DataFrame,
    time_col: str,
    n_splits: int,
    test_size: float,
    min_train_size: int | None = None,
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """Yield rolling-window train/test splits.

    Args:
        df: Input data.
        time_col: Timestamp column name.
        n_splits: Number of rolling splits.
        test_size: Fraction of data per test window.
        min_train_size: Optional minimum train window size.
    """
    if n_splits <= 0:
        raise ValueError("n_splits must be positive.")

    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)
    test_n = int(n * test_size)
    if test_n <= 0:
        raise ValueError("test_size is too small for dataset length.")

    max_splits = (n - 1) // test_n
    n_splits = min(n_splits, max_splits)
    if n_splits <= 0:
        raise ValueError("Not enough data for the requested splits.")

    for i in range(n_splits):
        test_end = n - i * test_n
        test_start = test_end - test_n
        train_end = test_start
        train_start = 0
        if min_train_size is not None:
            train_start = max(0, train_end - min_train_size)
        train = df_sorted.iloc[train_start:train_end].copy()
        test = df_sorted.iloc[test_start:test_end].copy()
        if len(train) == 0 or len(test) == 0:
            continue
        yield train, test
