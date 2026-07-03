"""
Auto-cleaner — detects and optionally fixes common data issues.

Functions:
    detect_issues(df) → list[dict]   — diagnostics (no modification)
    clean(df, strategy) → pd.DataFrame — return cleaned copy
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Issue detection — returns a list of {issue_type, column, severity, detail}
# ---------------------------------------------------------------------------
def detect_issues(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Scan *df* for common data problems. Returns a list of issue dicts."""
    issues: list[dict] = []

    # 1. Missing values (per column)
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        pct = n_missing / max(len(df), 1) * 100
        if n_missing > 0:
            severity = "high" if pct > 20 else ("medium" if pct > 5 else "low")
            issues.append({
                "issue_type": "missing_values",
                "column": col,
                "severity": severity,
                "detail": f"{n_missing} missing ({pct:.1f}%)",
            })

    # 2. Duplicate rows
    n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        pct = n_dupes / max(len(df), 1) * 100
        severity = "high" if pct > 5 else "medium"
        issues.append({
            "issue_type": "duplicate_rows",
            "column": "(entire row)",
            "severity": severity,
            "detail": f"{n_dupes} duplicate rows ({pct:.1f}%)",
        })

    # 3. Empty string names / labels in id-like columns
    for col in df.columns:
        if "id" in col.lower() or "name" in col.lower():
            empty = int(((df[col].astype(str) == "") | (df[col].astype(str) == "nan")).sum())
            if empty > 0:
                issues.append({
                    "issue_type": "empty_strings",
                    "column": col,
                    "severity": "low",
                    "detail": f"{empty} empty/NaN entries in label column",
                })

    # 4. Negative values in count-type columns
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            neg = int((df[col] < 0).sum())
            if neg > 0:
                issues.append({
                    "issue_type": "negative_values",
                    "column": col,
                    "severity": "medium",
                    "detail": f"{neg} negative values (may be invalid for counts)",
                })

    # 5. Columns with very low variance (>95% same value)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) > 1:
                iqr = non_null.quantile(0.75) - non_null.quantile(0.25)
                if iqr / max(abs(non_null.median()), 1e-9) < 0.01:
                    issues.append({
                        "issue_type": "low_variance",
                        "column": col,
                        "severity": "low",
                        "detail": f"Very low variance (possibly constant column)",
                    })

    # 6. High cardinality categoricals (>50% unique)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) > 10:
                uq_ratio = non_null.nunique() / len(non_null)
                if uq_ratio > 0.5:
                    issues.append({
                        "issue_type": "high_cardinality_categorical",
                        "column": col,
                        "severity": "low",
                        "detail": f"{uq_ratio*100:.0f}% unique values (may be mis-typed as categorical)",
                    })

    return issues


# ---------------------------------------------------------------------------
# Cleaning strategies
# ---------------------------------------------------------------------------
CLEAN_STRATEGIES = {
    "none": "Return the data as-is (diagnostics only).",
    "minimal": "Drop duplicates + rows with >50% missing; strip empty strings.",
    "moderate": "minimal + fill numeric NaN with median, categorical NaN with mode.",
    "aggressive": "moderate + drop columns with >30% missing + correct obvious type mismatches.",
}


def clean(
    df: pd.DataFrame,
    strategy: str = "moderate",
) -> pd.DataFrame:
    """Return a cleaned copy of *df* according to *strategy*.

    Supported strategies (see CLEAN_STRATEGIES for descriptions):
        none, minimal, moderate, aggressive
    """
    if strategy not in CLEAN_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}. Choices: {list(CLEAN_STRATEGIES)}")

    cleaned = df.copy()

    # Strip whitespace from all string columns
    for col in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    if strategy == "none":
        return cleaned

    # --- minimal ----------------------------------------------------------
    # Drop exact duplicate rows (keep first)
    cleaned = cleaned[~cleaned.duplicated(keep="first")].reset_index(drop=True)

    # Remove rows where >50% of cells are NaN
    thresh = 0.5 * len(cleaned.columns)
    cleaned = cleaned[cleaned.notna().sum(axis=1) >= thresh].reset_index(drop=True)

    if strategy == "minimal":
        return cleaned

    # --- moderate ---------------------------------------------------------
    # Fill numeric NaN with median per column
    for col in cleaned.select_dtypes(include=["number"]).columns:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    # Fill categorical NaN with mode (or "" if all NaN)
    for col in cleaned.select_dtypes(exclude=["number"]).columns:
        if cleaned[col].isna().any():
            mode_vals = cleaned[col].mode()
            fill_val = str(mode_vals.iloc[0]) if len(mode_vals) > 0 else ""
            cleaned[col] = cleaned[col].fillna(fill_val)

    if strategy == "moderate":
        return cleaned

    # --- aggressive -------------------------------------------------------
    # Drop columns with >30% missing values (after minimal cleaning)
    drop_cols = [
        col for col in cleaned.columns
        if cleaned[col].isna().sum() / len(cleaned) > 0.30
    ]
    if drop_cols:
        print(f"[cleaner] Dropping columns with >30% missing: {drop_cols}")
        cleaned = cleaned.drop(columns=drop_cols)

    # Fix obvious type mismatches — numeric columns stored as strings
    for col in cleaned.select_dtypes(include=["object"]).columns:
        if cleaned[col].isna().all():
            continue
        try:
            coerced = pd.to_numeric(cleaned[col])
            # If >80% successfully converted, trust it
            non_null_pct = cleaned[col].notna().sum() / max(len(cleaned), 1)
            if coerced.notna().sum() / max(len(cleaned), 1) > 0.8 and non_null_pct > 0.5:
                print(f"[cleaner] Coercing {col} from string → numeric")
                cleaned[col] = coerced
        except (ValueError, TypeError):
            pass

    return cleaned


if __name__ == "__main__":
    import pandas as pd
    from app.generator import generate_nba_dataset
    rows = generate_nba_dataset(50)
    df = pd.DataFrame(rows)

    issues = detect_issues(df)
    print("=== Detected Issues ===")
    for iss in issues:
        print(f"  [{iss['severity'].upper()}] {iss['issue_type']}: {iss['column']} — {iss['detail']}")

    cleaned = clean(df, strategy="moderate")
    issues_after = detect_issues(cleaned)
    print(f"\n=== After Cleaning (moderate) ===")
    for iss in issues_after:
        print(f"  [{iss['severity'].upper()}] {iss['issue_type']}: {iss['column']} — {iss['detail']}")
