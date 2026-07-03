# ChatData — Tests

"""Tests for ChatData modules.

Run with: pytest tests/ -v
"""

import pandas as pd
import numpy as np

from app.analyzer import profile, quick_stats
from app.cleaner import clean, detect_issues
from app.sandbox import safe_execute
from app.generator import generate_nba_dataset


# ════════════════════════════════════════════════════════════════════════════
# Dataset Generator
# ════════════════════════════════════════════════════════════════════════════
class TestDatasetGenerator:
    def test_generates_correct_row_count(self):
        rows = generate_nba_dataset(n_players=100, seed=42)
        assert len(rows) == 100

    def test_each_row_has_required_fields(self):
        rows = generate_nba_dataset(n_players=50, seed=42)
        required = ["PlayerID", "Name", "Team", "Position", "PointsPerGame"]
        for row in rows:
            for field in required:
                assert field in row, f"Row missing {field}"

    def test_positions_are_valid(self):
        rows = generate_nba_dataset(n_players=200, seed=42)
        valid = {"PG", "SG", "SF", "PF", "C"}
        for row in rows:
            assert row["Position"] in valid

    def test_teams_are_from_list(self):
        rows = generate_nba_dataset(n_players=500, seed=42)
        expected_teams = {
            "ATL", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET",
            "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL",
            "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR",
            "SAC", "SAS", "UTA", "WAS", "BKN", "ORB",
        }
        for row in rows:
            assert row["Team"] in expected_teams

    def test_deterministic_with_seed(self):
        r1 = generate_nba_dataset(n_players=50, seed=42)
        r2 = generate_nba_dataset(n_players=50, seed=42)
        assert r1 == r2

    def test_different_seeds_produce_different_data(self):
        r1 = generate_nba_dataset(n_players=50, seed=42)
        r2 = generate_nba_dataset(n_players=50, seed=99)
        # With 30 teams and positions, at least one player should differ
        assert any(r1[i] != r2[i] for i in range(len(r1)))


# ════════════════════════════════════════════════════════════════════════════
# Analyzer
# ════════════════════════════════════════════════════════════════════════════
class TestDataAnalyzer:
    def setup_method(self):
        rows = generate_nba_dataset(n_players=30, seed=42)
        self.df = pd.DataFrame(rows)

    def test_profile_returns_expected_keys(self):
        prof = profile(self.df)
        expected_keys = {
            "total_columns", "total_rows", "memory_bytes", "column_stats",
            "missing", "duplicates", "outliers", "value_counts", "sample_rows",
        }
        assert set(prof.keys()) == expected_keys

    def test_profile_row_count_matches(self):
        prof = profile(self.df)
        assert prof["total_rows"] == len(self.df)
        assert prof["total_columns"] == len(self.df.columns)

    def test_sample_rows_not_empty(self):
        prof = profile(self.df)
        assert len(prof["sample_rows"]) > 0
        assert len(prof["sample_rows"]) <= 20  # SAMPLE_SIZE

    def test_quick_stats_returns_text(self):
        qs = quick_stats(self.df)
        assert "stats_text" in qs
        assert len(qs["stats_text"]) > 0

    def test_detects_missing_in_nba_dataset(self):
        prof = profile(self.df)
        # Generator intentionally leaves ~3% of Ages blank (None → NaN)
        has_blank_age = self.df["Age"].isna().any()
        if has_blank_age:
            assert "Age" in prof["missing"]

    def test_detects_duplicates_when_present(self):
        duped = pd.concat([self.df, self.df.iloc[:5]], ignore_index=True)
        prof = profile(duped)
        assert prof["duplicates"] >= 5


# ════════════════════════════════════════════════════════════════════════════
# Cleaner
# ════════════════════════════════════════════════════════════════════════════
class TestDataCleaner:
    def setup_method(self):
        rows = generate_nba_dataset(n_players=30, seed=42)
        self.df = pd.DataFrame(rows)

    def test_clean_none_returns_copy(self):
        cleaned = clean(self.df, strategy="none")
        assert len(cleaned) == len(self.df)
        assert set(cleaned.columns) == set(self.df.columns)

    def test_clean_minimal_drops_exact_duplicates(self):
        # Create a DataFrame with exact duplicates
        dup_df = pd.concat([self.df, self.df.iloc[:3]], ignore_index=True)
        before = len(dup_df)
        cleaned = clean(dup_df, strategy="minimal")
        after = len(cleaned)
        assert after == before - 3

    def test_clean_moderate_fills_missing_numeric(self):
        df_with_na = self.df.copy()
        df_with_na.loc[0, "PointsPerGame"] = np.nan
        cleaned = clean(df_with_na, strategy="moderate")
        assert cleaned.loc[0, "PointsPerGame"] is not None and not pd.isna(cleaned.loc[0, "PointsPerGame"])

    def test_clean_aggressive_drops_high_missing_cols(self):
        df_many_na = self.df.copy()
        # Fill an entire column with NaN
        df_many_na["BrandNewColumn"] = np.nan
        cleaned = clean(df_many_na, strategy="aggressive")
        assert "BrandNewColumn" not in cleaned.columns

    def test_clean_invalid_strategy_raises(self):
        try:
            clean(self.df, strategy="invalid_strategy_name_xyz")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # expected

    def test_detect_issues_returns_list(self):
        issues = detect_issues(self.df)
        assert isinstance(issues, list)
        for issue in issues:
            assert "issue_type" in issue
            assert "column" in issue
            assert "severity" in issue
            assert issue["severity"] in ("high", "medium", "low")


# ════════════════════════════════════════════════════════════════════════════
# Sandbox (critical — this is what makes the app safe)
# ════════════════════════════════════════════════════════════════════════════
class TestSandbox:
    def setup_method(self):
        self.df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": ["x", "y", "z", "w", "v"],
        })

    def test_valid_expression_succeeds(self):
        result = safe_execute("df['a'].mean()", self.df)
        assert result["success"]
        assert result["result"] == 3.0

    def test_blocked_import(self):
        result = safe_execute("__import__('os').listdir('.')", self.df)
        assert not result["success"]

    def test_blocked_open(self):
        result = safe_execute("open('/etc/passwd')", self.df)
        assert not result["success"]

    def test_blocked_exec(self):
        result = safe_execute("exec('print(1)')", self.df)
        assert not result["success"]

    def test_blocked_builtins(self):
        result = safe_execute("__builtins__['eval']('__import__(\"os\")')", self.df)
        assert not result["success"]

    def test_bad_column_name(self):
        result = safe_execute("df['nonexistent'].sum()", self.df)
        assert not result["success"]
        assert "KeyError" in result["error"] or "not found" in result["error"].lower()

    def test_empty_expression(self):
        result = safe_execute("", self.df)
        assert not result["success"]

    def test_none_expression(self):
        result = safe_execute(None, self.df)
        assert not result["success"]

    def test_returns_code_always(self):
        result = safe_execute("df['a'].sum()", self.df)
        assert "code" in result
        assert result["code"] == "df['a'].sum()"

    def test_safe_builtins_work(self):
        result = safe_execute("max(df['b'])", self.df)
        assert result["success"]
        assert result["result"] == 50

    def test_filtering_works(self):
        result = safe_execute("df[df['a'] > 2]['c'].tolist()", self.df)
        assert result["success"]
        assert set(result["result"]) == {"z", "w", "v"}


# ════════════════════════════════════════════════════════════════════════════
# Integration: analyzer + cleaner round-trip
# ════════════════════════════════════════════════════════════════════════════
class TestIntegration:
    def test_full_pipeline(self):
        """Simulate upload → profile → clean → profile."""
        rows = generate_nba_dataset(n_players=30, seed=42)
        df = pd.DataFrame(rows)

        # Profile before cleaning
        prof_before = profile(df)

        # Clean
        cleaned = clean(df, strategy="moderate")

        # Profile after cleaning
        prof_after = profile(cleaned)

        assert prof_after["total_rows"] >= prof_before["total_rows"] - 1
        # Missing should not explode (cleaning may introduce some from NaN handling)
        assert prof_after["total_columns"] == prof_before["total_columns"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
