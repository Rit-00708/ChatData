"""
Data analyzer — profiles datasets and detects issues.

Functions:
    profile(df) → dict  — full auto-profile with stats + issue detection
    quick_stats(df)   → dict  — lightweight summary (for passing to LLM)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from .config import SAMPLE_SIZE

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def profile(df: pd.DataFrame) -> dict[str, Any]:
    """Return a comprehensive auto-profile of the DataFrame.

    Keys in the returned dict:
        column_stats     — per-column dtype / counts / describe summary
        missing          — {col: count} for every column with > 0 nulls
        duplicates       — int; number of duplicate rows
        outliers         — {col: [list of outlier values]} (IQR method)
        value_counts     — {col: [{value, count}, ...]} top-10 mode per cat col
        sample_rows      — list-of-dicts for first *SAMPLE_SIZE* rows
        total_columns    — int
        total_rows       — int
        memory_bytes     — int (df.memory_usage().sum())
    """
    n_rows, n_cols = df.shape

    # --- column stats per dtype -------------------------------------------
    column_stats: list[dict] = []
    for col in df.columns:
        info: dict[str, Any] = {"column": col, "dtype": str(df[col].dtype), "non_null": int(df[col].notna().sum())}

        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            info["mean"] = round(float(desc["mean"]), 2)
            info["std"] = round(float(desc["std"]), 2)
            info["min"] = float(desc["min"])
            info["max"] = float(desc["max"])
            info["median"] = round(float(desc["50%"]), 2)
        else:
            vc = df[col].value_counts().head(10)
            info["unique_count"] = int(df[col].nunique())
            info["top_values"] = [
                {"value": str(v), "count": int(c)} for v, c in zip(vc.index, vc.values)
            ]

        column_stats.append(info)

    # --- missing values ---------------------------------------------------
    missing = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()}

    # --- duplicates --------------------------------------------------------
    dupes = int(df.duplicated().sum())

    # --- outliers (IQR method, numeric only) -------------------------------
    outliers: dict[str, list] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        out_vals = df.loc[(df[col] < lower) | (df[col] > upper), col].dropna()
        if len(out_vals) > 0:
            outliers[col] = [round(float(v), 2) for v in sorted(out_vals.tolist())]

    # --- value counts (categoricals only, top-10) --------------------------
    cat_value_counts: dict[str, list[dict]] = {}
    for col in df.select_dtypes(exclude=[np.number]).columns:
        vc = df[col].dropna().value_counts().head(10)
        if len(vc) > 0:
            cat_value_counts[col] = [
                {"value": str(v), "count": int(c)} for v, c in zip(vc.index, vc.values)
            ]

    # --- sample rows -------------------------------------------------------
    sample = df.head(SAMPLE_SIZE).to_dict(orient="records")

    return {
        "total_columns": n_cols,
        "total_rows": n_rows,
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
        "column_stats": column_stats,
        "missing": missing,
        "duplicates": dupes,
        "outliers": outliers,
        "value_counts": cat_value_counts,
        "sample_rows": sample,
    }


def quick_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Lightweight summary — ideal for sending to an LLM as context.

    Returns a string with: dtypes, describe() for numeric, top-5 value-counts
    for categoricals, missing/duplicates counts, and first 5 sample rows.
    """
    lines = [f"Shape: {df.shape[0]} rows × {df.shape[1]} columns"]
    lines.append("\nDtypes:")
    for col, dtype in df.dtypes.items():
        lines.append(f"  {col}: {dtype}")

    # Numeric describe (abbreviated)
    nums = df.select_dtypes(include=[np.number])
    if not nums.empty:
        desc = nums.describe().round(2)
        lines.append("\nDescribe (numeric):")
        for col in desc.columns:
            stats_str = ", ".join(f"{k}={desc.loc[k, col]:.2f}" for k in desc.index)
            lines.append(f"  {col}: [{stats_str}]")

    # Categoricals top-5 value counts
    cats = df.select_dtypes(exclude=[np.number])
    for col in cats.columns:
        vc = df[col].dropna().value_counts().head(5)
        if len(vc) > 0:
            lines.append(f"\nTop values for {col}:")
            for v, c in zip(vc.index, vc.values):
                lines.append(f"  {v}: {c}")

    # Missing & duplicates
    n_missing = int(df.isna().sum().sum())
    n_dupes = int(df.duplicated().sum())
    if n_missing > 0 or n_dupes > 0:
        lines.append("\nMissing values: " + str(n_missing))
        lines.append("Duplicate rows: " + str(n_dupes))

    # Sample rows
    lines.append("\nFirst 5 sample rows:")
    for _, row in df.head(5).iterrows():
        vals = ", ".join(f"{c}={row[c]}" for c in df.columns[:10])
        lines.append(f"  {{ {vals} }}")

    return {"stats_text": "\n".join(lines)}


if __name__ == "__main__":
    import pandas as pd
    from app.generator import generate_nba_dataset
    rows = generate_nba_dataset(30)
    df = pd.DataFrame(rows)
    prof = profile(df)
    print(f"Rows: {prof['total_rows']}, Columns: {prof['total_columns']}")
    print(f"Missing values in columns: {prof['missing']}")
    print(f"Duplicate rows: {prof['duplicates']}")
    print(f"Outlier columns: {list(prof['outliers'].keys())}")
