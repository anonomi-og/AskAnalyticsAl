# viz.py
"""
Very simple heuristic to choose a Streamlit visual for a pandas DataFrame.
"""

# viz.py
import streamlit as st
import pandas as pd
import altair as alt

def choose_visual(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("No table output to visualise.")
        return

    # --- 1-cell KPI ---------------------------------------------------
    if df.shape == (1, 1):
        label = df.columns[0] if len(df.columns) else "Result"
        st.metric(label=label, value=df.iat[0, 0])
        return

    # --- datetime → line chart ---------------------------------------
    if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
        st.line_chart(df.set_index(df.columns[0]))
        return

    # --- 2-col category → bar chart ----------------------------------
    if df.shape[1] == 2 and df.dtypes.iloc[0] == "object":
        st.bar_chart(df.set_index(df.columns[0]))
        return

    # --- 2-col numeric → scatter -------------------------------------
    if df.shape[1] == 2 and all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes):
        st.scatter_chart(df)
        return

    # --- fallback table ----------------------------------------------
    st.dataframe(df, use_container_width=True)
