"""Unit tests for src.utils module."""

import pytest
import pandas as pd
import numpy as np

from src.utils import TimeSplit, time_train_test_split, rolling_window_cv


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample time-series DataFrame for testing."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
        "value": np.random.randn(100),
        "category": ["A", "B"] * 50,
    })


class TestTimeSplit:
    """Tests for TimeSplit dataclass."""

    def test_time_split_creation(self, sample_dataframe: pd.DataFrame) -> None:
        """Test TimeSplit can be instantiated."""
        train = sample_dataframe.iloc[:70]
        test = sample_dataframe.iloc[70:]
        split = TimeSplit(train=train, test=test)
        assert len(split.train) == 70
        assert len(split.test) == 30

    def test_time_split_immutable(self, sample_dataframe: pd.DataFrame) -> None:
        """Test TimeSplit is immutable (frozen)."""
        train = sample_dataframe.iloc[:70]
        test = sample_dataframe.iloc[70:]
        split = TimeSplit(train=train, test=test)
        with pytest.raises(AttributeError):
            split.train = sample_dataframe


class TestTimeTrainTestSplit:
    """Tests for time_train_test_split function."""

    def test_basic_split(self, sample_dataframe: pd.DataFrame) -> None:
        """Test basic time-ordered train/test split."""
        split = time_train_test_split(sample_dataframe, "timestamp", test_size=0.3)
        assert len(split.train) == 70
        assert len(split.test) == 30
        assert len(split.train) + len(split.test) == len(sample_dataframe)

    def test_split_maintains_order(self, sample_dataframe: pd.DataFrame) -> None:
        """Test split maintains chronological order."""
        split = time_train_test_split(sample_dataframe, "timestamp", test_size=0.3)
        train_max = split.train["timestamp"].max()
        test_min = split.test["timestamp"].min()
        assert train_max < test_min

    def test_split_preserves_data(self, sample_dataframe: pd.DataFrame) -> None:
        """Test split doesn't lose or duplicate data."""
        split = time_train_test_split(sample_dataframe, "timestamp", test_size=0.3)
        combined = pd.concat([split.train, split.test], ignore_index=False).sort_index()
        pd.testing.assert_frame_equal(
            combined.reset_index(drop=True),
            sample_dataframe.reset_index(drop=True),
        )

    def test_invalid_test_size_too_small(self, sample_dataframe: pd.DataFrame) -> None:
        """Test ValueError for test_size <= 0."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            time_train_test_split(sample_dataframe, "timestamp", test_size=0.0)

    def test_invalid_test_size_too_large(self, sample_dataframe: pd.DataFrame) -> None:
        """Test ValueError for test_size >= 1."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            time_train_test_split(sample_dataframe, "timestamp", test_size=1.0)

    def test_invalid_test_size_empty_split(self) -> None:
        """Test ValueError when split results in empty partition."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=2),
            "value": [1, 2],
        })
        with pytest.raises(ValueError, match="empty train or test split"):
            time_train_test_split(df, "timestamp", test_size=0.99)


class TestRollingWindowCV:
    """Tests for rolling_window_cv function."""

    def test_rolling_window_basic(self, sample_dataframe: pd.DataFrame) -> None:
        """Test rolling window CV generates expected splits."""
        splits = list(rolling_window_cv(
            sample_dataframe,
            time_col="timestamp",
            n_splits=3,
            test_size=0.2,
        ))
        assert len(splits) == 3
        for train, test in splits:
            assert len(train) > 0
            assert len(test) > 0

    def test_rolling_window_splits_dont_overlap(
        self,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Test rolling windows don't overlap temporally."""
        splits = list(rolling_window_cv(
            sample_dataframe,
            time_col="timestamp",
            n_splits=3,
            test_size=0.2,
        ))
        for train, test in splits:
            assert train["timestamp"].max() < test["timestamp"].min()

    def test_rolling_window_test_set_size(self, sample_dataframe: pd.DataFrame) -> None:
        """Test rolling window test sets have expected size."""
        splits = list(rolling_window_cv(
            sample_dataframe,
            time_col="timestamp",
            n_splits=3,
            test_size=0.2,
        ))
        for _, test in splits:
            assert len(test) == int(0.2 * len(sample_dataframe))

    def test_rolling_window_with_min_train_size(
        self,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Test rolling window respects minimum train size."""
        min_size = 50
        splits = list(rolling_window_cv(
            sample_dataframe,
            time_col="timestamp",
            n_splits=2,
            test_size=0.2,
            min_train_size=min_size,
        ))
        for train, _ in splits:
            assert len(train) >= min_size

    def test_rolling_window_single_split(self, sample_dataframe: pd.DataFrame) -> None:
        """Test rolling window with n_splits=1."""
        splits = list(rolling_window_cv(
            sample_dataframe,
            time_col="timestamp",
            n_splits=1,
            test_size=0.3,
        ))
        assert len(splits) == 1
