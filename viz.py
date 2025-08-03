# viz.py
"""
Very simple heuristic to choose a Streamlit visual for a pandas DataFrame.
"""

import streamlit as st
import pandas as pd
import altair as alt


def choose_visual(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("No table output to visualise.")
        return

    # 1-cell result → KPI
    if df.size == 1:
        st.metric(label=df.columns[0] if df.columns else "Result",
                  value=df.iloc[0, 0])
        return

    # Time-series line chart if first column looks like a date
    if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
        st.line_chart(df.set_index(df.columns[0]))
        return

    # 2-column categorical → bar
    if df.shape[1] == 2 and df.dtypes.iloc[0] == "object":
        st.bar_chart(df.set_index(df.columns[0]))
        return

    # Scatter for two numeric cols
    if df.shape[1] == 2 and all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes):
        st.scatter_chart(df)
        return

    # Fallback – interactive table
    st.dataframe(df, use_container_width=True)
