"""
Exploratory Data Analysis — auto-generates charts and summary statistics.

Functions:
    generate_eda_charts(df) → list[PlotlyFigure | Figure]  — one chart per numeric col + extras
    render_charts(figures, engine='plotly') → None         — Streamlit-compatible rendering
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Chart generation (returns Plotly figures or matplotlib Figures)
# ---------------------------------------------------------------------------
def _get_chart_engine():
    """Lazy-import the chart engine to avoid circular deps."""
    from app import config
    return config.CHART_ENGINE


def generate_eda_charts(df: pd.DataFrame, max_cols: int = 20) -> list[Any]:
    """Generate a list of EDA figures — one per numeric column + extras.

    Returns figures compatible with the configured CHART_ENGINE (Plotly or matplotlib).
    At most *max_cols* numeric columns get individual charts.
    Always includes:
        1. Correlation heatmap (for all numeric cols, up to max_cols)
        2. Distribution plot per numeric column (up to 8)
        3. Bar chart for top categorical column with <20 unique values
    """
    figures = []
    engine = _get_chart_engine()

    # --- correlation heatmap ----------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        n_heatmap = min(len(numeric_cols), max_cols)
        corr_df = df[numeric_cols[:n_heatmap]].corr(method="pearson")
        figures.append(_make_correlation_heatmap(corr_df, engine))

    # --- distributions for each numeric column (up to 8) ------------------
    for col in numeric_cols[:min(8, len(numeric_cols))]:
        figures.append(_make_distribution(col, df, engine))

    # --- bar chart for the most interesting categorical column ---------------
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in cat_cols:
        non_null = df[col].dropna()
        n_unique = non_null.nunique()
        # Pick a col with 2-15 unique values (good bar-chart range)
        if 2 <= n_unique <= 15:
            vc = non_null.value_counts().head(10)
            figures.append(_make_bar_chart(col, vc, engine))
            break  # just one cat bar chart
    else:
        # Fallback: if no good categorical col, show team/position distribution
        for col in cat_cols:
            non_null = df[col].dropna()
            vc = non_null.value_counts().head(10)
            figures.append(_make_bar_chart(col, vc, engine))
            break

    return figures


# ---------------------------------------------------------------------------
# Individual chart helpers
# ---------------------------------------------------------------------------
def _make_correlation_heatmap(corr: pd.DataFrame, engine: str) -> Any:
    """Correlation heatmap for numeric columns."""
    if engine == "plotly":
        import plotly.graph_objects as go
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                hovertemplate="%{y} vs %{x}<br>r = %{z:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Correlation Heatmap",
            height=600,
            margin=dict(l=80, r=80, t=40, b=80),
        )
        return fig
    else:
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", ax=ax, vmin=-1, vmax=1)
        ax.set_title("Correlation Heatmap", fontsize=14)
        return fig


def _make_distribution(col: str, df: pd.DataFrame, engine: str) -> Any:
    """Histogram / KDE for a numeric column."""
    data = df[col].dropna()
    if len(data) < 2:
        return None

    if engine == "plotly":
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data, nbinsx=min(50, max(10, int(len(data) ** 0.5))),
                                   name=col, marker_color="steelblue", opacity=0.7))
        # Overlay KDE via numpy + line trace
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(data.values)
            xs = np.linspace(data.min(), data.max(), 200)
            fig.add_trace(go.Scatter(x=xs, y=kde(xs), mode="lines", name="KDE",
                                     line=dict(color="red", width=2)))
        except Exception:
            pass
        fig.update_layout(
            title=f"Distribution of {col}",
            xaxis_title=col,
            yaxis_title="Count",
            height=400,
            showlegend=False,
        )
        return fig
    else:
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data, kde=True, ax=ax, bins=min(50, max(10, int(len(data) ** 0.5))),
                     color="steelblue", alpha=0.7)
        ax.set_title(f"Distribution of {col}", fontsize=12)
        return fig


def _make_bar_chart(col: str, vc: pd.Series, engine: str) -> Any:
    """Bar chart for a categorical column."""
    if engine == "plotly":
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=vc.index.astype(str),
            y=vc.values,
            marker_color="teal",
            opacity=0.8,
        ))
        fig.update_layout(
            title=f"Value Counts of {col}",
            xaxis_title=col,
            yaxis_title="Count",
            height=min(500, len(vc) * 40 + 100),
        )
        return fig
    else:
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(max(6, len(vc) * 0.5), 4))
        sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax, color="teal", alpha=0.8)
        ax.set_title(f"Value Counts of {col}", fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        return fig


# ---------------------------------------------------------------------------
# Summary text — auto-generated insights from basic stats
# ---------------------------------------------------------------------------
def generate_summary_text(df: pd.DataFrame) -> str:
    """Auto-generate a short human-readable summary of the dataset.

    This is useful for displaying EDA results without charts, or as context
    before the LLM generates its own insights.
    """
    n_rows, n_cols = df.shape
    lines = [
        f"## Dataset Summary",
        f"**{n_rows:,} rows × {n_cols} columns**",
        f"",
        f"### Data Types",
    ]

    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        lines.append(f"- **{dtype}**: {count} columns")

    # Numeric summaries
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        lines.append(f"\n### Numeric Column Highlights")
        desc = df[numeric_cols].describe().round(2)
        for col in numeric_cols[:10]:  # limit to first 10
            mean_val = desc.loc["mean", col]
            std_val = desc.loc["std", col]
            min_val = desc.loc["min", col]
            max_val = desc.loc["max", col]
            lines.append(
                f"- **{col}**: mean={mean_val:.1f}, std={std_val:.1f}, "
                f"range=[{min_val:.1f}, {max_val:.1f}]"
            )

    # Missing values
    missing = df.isna().sum()
    high_missing = missing[missing > 0].sort_values(ascending=False)
    if len(high_missing) > 0:
        lines.append(f"\n### Columns with Missing Values")
        for col, count in high_missing.head(5).items():
            pct = count / n_rows * 100
            lines.append(f"- **{col}**: {count} missing ({pct:.1f}%)")

    # Duplicate rows
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        lines.append(f"\n- **Duplicate rows**: {int(n_dupes)}")

    return "\n".join(lines)


if __name__ == "__main__":
    import pandas as pd
    from app.generator import generate_nba_dataset
    rows = generate_nba_dataset(30)
    df = pd.DataFrame(rows)
    charts = generate_eda_charts(df)
    print(f"Generated {len(charts)} EDA charts")
    summary = generate_summary_text(df)
    print(summary)
