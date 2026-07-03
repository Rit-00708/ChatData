"""
ChatData — A Local LLM-Powered Data Analysis Assistant.

Streamlit single-page app with:
  - Sidebar: file upload, cleaning strategy selector, LLM provider toggle, chart engine toggle
  - Main area: EDA panel (auto profile + charts), chat-style Q&A panel, insights panel
"""

from __future__ import annotations

import io
import os
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# ── App modules (all local) ────────────────────────────────────────────────
from app.analyzer import profile, quick_stats
from app.cleaner import clean, detect_issues
from app.eda import generate_eda_charts, generate_summary_text
from app.llm_client import chat, get_insights, generate_pandas_expr
from app.sandbox import format_result, safe_execute

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChatData — Talk to Your Data",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar state helpers (Streamlit reruns everything, so we persist via session_state) ──
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None  # the uploaded DataFrame (before cleaning)
if "df_cleaned" not in st.session_state:
    st.session_state.df_cleaned = None  # after cleaning
if "uploaded_path" not in st.session_state:
    st.session_state.uploaded_path = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history for the Q&A panel
if "profile" not in st.session_state:
    st.session_state.profile = None  # auto-profile result


# ════════════════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Settings")

    # ── File upload ────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"],
        key="file_uploader",
    )

    if uploaded is not None:
        st.session_state.uploaded_path = uploaded.name
        try:
            if uploaded.name.lower().endswith(".csv"):
                st.session_state.df_raw = pd.read_csv(uploaded)
            else:
                st.session_state.df_raw = pd.read_excel(uploaded)
            st.success(f"Loaded ✅ {uploaded.name} ({st.session_state.df_raw.shape[0]:,} rows)")
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            st.stop()

    # ── Clean button (if data loaded) ──────────────────────────────────────
    if st.session_state.df_raw is not None:
        st.divider()
        strategy = st.selectbox(
            "Cleaning strategy",
            ["none", "minimal", "moderate", "aggressive"],
            help="How aggressively to auto-clean the data before analysis.",
        )

        if st.button("🧹 Apply Cleaning", type="primary"):
            try:
                st.session_state.df_cleaned = clean(
                    st.session_state.df_raw, strategy=strategy
                )
                # Recalculate profile on cleaned data
                st.session_state.profile = profile(st.session_state.df_cleaned)
                st.rerun()
            except Exception as e:
                st.error(f"Cleaning failed: {e}")

    # ── LLM provider ───────────────────────────────────────────────────────
    st.divider()
    llm_provider = st.selectbox(
        "LLM Provider",
        ["ollama", "groq", "fake"],
        help=(
            "ollama — local Ollama (default). "
            "groq — Groq cloud API (set GROQ_API_KEY env var). "
            "fake — canned responses for testing without a real LLM."
        ),
    )
    os.environ["LLM_PROVIDER"] = llm_provider

    # ── Chart engine ───────────────────────────────────────────────────────
    chart_engine = st.selectbox(
        "Chart Engine",
        ["plotly", "matplotlib"],
        help="Plotly gives interactive charts; matplotlib is lighter weight.",
    )
    os.environ["CHART_ENGINE"] = chart_engine

    # ── Model name (if ollama) ─────────────────────────────────────────────
    if llm_provider == "ollama":
        model_name = st.text_input(
            "Ollama Model",
            value=os.environ.get("OLLAMA_MODEL", "qwen2.5:32b"),
            help='e.g. "qwen2.5:32b" or "llama3:8b"',
        )

    # ── Reset button ───────────────────────────────────────────────────────
    st.divider()
    if st.button("🗑️ Clear Data", type="secondary"):
        st.session_state.df_raw = None
        st.session_state.df_cleaned = None
        st.session_state.uploaded_path = None
        st.session_state.profile = None
        st.session_state.messages = []
        st.rerun()

    # ── Status / help box ──────────────────────────────────────────────────
    with st.expander("ℹ️ How it works"):
        st.caption(textwrap.dedent("""\
            1. Upload a CSV/Excel file → auto-profile & cleaning
            2. Browse the EDA panel (stats, charts)
            3. Ask a question → LLM writes pandas code → sandbox runs it
            4. Click "Generate Insights" for auto-summary"""))


# ════════════════════════════════════════════════════════════════════════════
# Main area — tabs
# ════════════════════════════════════════════════════════════════════════════

if st.session_state.df_raw is None:
    # ── Landing screen (no data loaded) ────────────────────────────────────
    st.title("📊 ChatData")
    st.caption(
        "A Local LLM-Powered Data Analysis Assistant — upload any CSV/Excel dataset "
        "and ask questions in plain English."
    )
    st.divider()

    # Load sample data button
    if st.button("Load Sample NBA Dataset", type="primary"):
        try:
            from app.generator import generate_nba_dataset
            n_rows = 450
            rows = generate_nba_dataset(n_players=n_rows, seed=42)
            df_sample = pd.DataFrame(rows)
            # Fix Age column — ensure no NaN for Streamlit display
            if "Age" in df_sample.columns:
                df_sample["Age"] = pd.to_numeric(df_sample["Age"], errors="coerce")
                df_sample["Age"] = df_sample["Age"].fillna(df_sample["Age"].median()).astype(int)
            st.session_state.df_raw = df_sample
            st.session_state.df_cleaned = df_sample.copy()
            st.session_state.uploaded_path = "nba_2024_25_players.csv (sample)"
            st.session_state.profile = profile(st.session_state.df_cleaned)
            st.rerun()
        except Exception as e:
            st.error(f"Failed to generate sample data: {e}")

    st.divider()
    with st.expander("📋 Or upload your own"):
        st.markdown("""
            - **CSV**: Standard comma-separated values file
            - **Excel**: `.xlsx` or `.xls` files
            - The app handles missing values, duplicates, and type detection automatically
            - Your data stays on your machine — nothing is uploaded to any server""")


else:
    df = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df_raw

    tabs = st.tabs(["📊 EDA", "💬 Q&A", "💡 Insights"])

    # ═══════════════════════════════════════════════════════════════════
    # TAB 1: EDA
    # ═══════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.header("📊 Exploratory Data Analysis")

        # Auto-profile summary
        if st.session_state.profile is None:
            st.session_state.profile = profile(df)

        prof = st.session_state.profile

        # --- Dataset overview row --------------------------------------------
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Rows", f"{prof['total_rows']:,}")
        with c2:
            st.metric("Columns", prof["total_columns"])
        with c3:
            st.metric("Missing Values", f"{sum(prof['missing'].values()):,}")
        with c4:
            st.metric("Duplicate Rows", prof["duplicates"])

        st.divider()

        # --- Auto-generated summary text -------------------------------------
        with st.expander("📝 Data Summary", expanded=True):
            st.markdown(generate_summary_text(df))

        # --- Issues detected -------------------------------------------------
        if prof.get("missing") and any(prof["missing"].values()):
            issues = detect_issues(df)
            with st.expander(f"⚠️ Detected Issues ({len(issues)} found)", expanded=bool(issues)):
                for iss in issues[:10]:  # cap display
                    icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(iss["severity"], "⚪")
                    st.caption(f"{icon} **{iss['issue_type']}** — {iss['column']}: {iss['detail']}")

        # --- Outliers --------------------------------------------------------
        if prof.get("outliers"):
            outlier_cols = list(prof["outliers"].keys())
            with st.expander(f"🔍 Outliers Detected ({len(outlier_cols)} columns)", expanded=True):
                for col, vals in prof["outliers"].items():
                    st.write(f"**{col}**: {len(vals)} outlier(s) — values: {vals[:10]}{'...' if len(vals) > 10 else ''}")

        # --- Charts ----------------------------------------------------------
        st.divider()
        st.subheader("📈 Charts")
        charts = generate_eda_charts(df)
        n_cols_chart = min(2, max(1, len(charts)))
        cols = st.columns(n_cols_chart)
        for i, fig in enumerate(charts):
            if fig is None:
                continue
            with cols[i % n_cols_chart]:
                st.plotly_chart(fig, use_container_width=True, key=f"eda_chart_{i}")

    # ═══════════════════════════════════════════════════════════════════
    # TAB 2: Q&A (chat-style)
    # ═══════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.header("💬 Ask a Question About Your Data")

        # Show the schema context (for transparency)
        with st.expander("🔧 Schema & Sample Rows (sent to LLM)", expanded=False):
            quick = quick_stats(df)
            st.text(quick["stats_text"])

        # Chat messages (persisted across reruns via session_state)
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input box
        if prompt := st.chat_input("What would you like to know about this data?"):
            # --- User message ---
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # --- Build schema context for LLM ---
            schema_lines = [f"Shape: {df.shape[0]} rows × {df.shape[1]} columns"]
            schema_lines.append("\nDtypes:")
            for col, dtype in df.dtypes.items():
                schema_lines.append(f"  {col}: {dtype}")

            # Add value counts for top categoricals
            for col in df.select_dtypes(exclude=[np.number]).columns[:3]:
                vc = df[col].dropna().value_counts().head(5)
                if len(vc) > 0:
                    schema_lines.append(f"\nTop values for {col}:")
                    for v, c in zip(vc.index, vc.values):
                        schema_lines.append(f"  {v}: {c}")

            schema_text = "\n".join(schema_lines)

            # --- Ask LLM to generate pandas code ---
            expr = None
            result = {"success": False, "error": "", "result": None}

            with st.spinner("Generating answer..."):
                try:
                    expr = generate_pandas_expr(prompt, schema_text)
                except RuntimeError as e:
                    st.error(str(e))
                    st.stop()

            if expr is None:
                st.warning("LLM returned no response. Try a different provider or rephrase your question.")
                st.session_state.messages.append(
                    {"role": "assistant", "content": "I wasn't able to generate an answer. Please try a different question."}
                )
                st.rerun()

            if expr.strip() == "UNANSWERABLE":
                st.warning("The LLM says this question can't be answered from the data provided.")
                st.session_state.messages.append(
                    {"role": "assistant", "content": "I couldn't find a way to answer that question with the data you've provided."}
                )
                st.rerun()

            # --- Execute in sandbox ---
            result = safe_execute(expr, df)

            if not result["success"]:
                answer = (
                    f"**Generated code:**\n```python\n{expr}\n```\n\n"
                    f"**Error:** {result['error']}"
                )
            else:
                # --- Check if result is plottable ---
                chart_fig = None
                if isinstance(result["result"], pd.Series):
                    if len(result["result"]) > 0 and len(result["result"]) <= 50:
                        try:
                            import plotly.graph_objects as go
                            chart_fig = go.Figure(go.Bar(
                                x=result["result"].index.astype(str),
                                y=result["result"].values,
                                marker_color="steelblue",
                                opacity=0.8,
                            ))
                            chart_fig.update_layout(
                                title=f"Result of: {prompt}",
                                xaxis_title=result["result"].index.name or "Categories",
                                yaxis_title="Value",
                                height=min(400, len(result["result"]) * 30 + 80),
                            )
                        except Exception:
                            pass

                # --- Format answer ---
                text_answer = format_result(result)
                answer = f"**Generated code:**\n```python\n{expr}\n```\n\n**Answer:**\n{text_answer}"

                if chart_fig:
                    with st.expander("📊 Show chart", expanded=True):
                        st.plotly_chart(chart_fig, use_container_width=True)

            st.session_state.messages.append({"role": "assistant", "content": answer})

        # Clear chat button
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()

    # ═══════════════════════════════════════════════════════════════════
    # TAB 3: Insights (auto-generated)
    # ═══════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.header("💡 Auto-Generated Insights")

        if quick := quick_stats(df):
            stats_text = quick["stats_text"]

            if "auto_insights" not in st.session_state:
                st.session_state.auto_insights = None

            col_btn, _ = st.columns([1, 3])
            with col_btn:
                if st.button("🧠 Generate Insights", type="primary", use_container_width=True):
                    with st.spinner("Sending data to LLM for analysis..."):
                        try:
                            insights = get_insights(stats_text)
                        except RuntimeError as e:
                            st.error(str(e))
                            st.stop()
                    st.session_state.auto_insights = insights

            if st.session_state.auto_insights:
                with st.container():
                    st.markdown(st.session_state.auto_insights)

            # Also show the EDA summary for reference
            with st.expander("📊 View Dataset Summary (not LLM-generated)", expanded=False):
                st.markdown(generate_summary_text(df))


# ════════════════════════════════════════════════════════════════════════════
# Footer
# ════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "ChatData — Runs entirely on your machine with Ollama. No data leaves your computer. "
    "For the public demo, set `LLM_PROVIDER=groq` + add a `GROQ_API_KEY`."
)
