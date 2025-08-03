import streamlit as st
from agents import get_agent
from viz import choose_visual
import pandas as pd

st.set_page_config(page_title="AI Data Assistant", layout="wide")
st.title("Ask your customer data")

question = st.text_input(
    "Natural-language question",
    placeholder="e.g. Show me a breakdown of contract types",
)

if st.button("Run", type="primary") and question:
    # ── Run agent ──────────────────────────────────────────────────
    with st.spinner("Thinking…"):
        agent  = get_agent()
        result = agent.invoke({"input": question})

    # ── Debug: which tools were used? ──────────────────────────────
    st.text(f"🛠️  Tools used: {[act.tool for act, _ in result['intermediate_steps']]}")

    # ── Natural-language answer ────────────────────────────────────
    st.success(result["output"])

    # ── Collect ALL DataFrames returned by query_with_df ───────────
    dfs = [
        obs
        for act, obs in result["intermediate_steps"]
        if act.tool == "query_with_df" and isinstance(obs, pd.DataFrame)
    ]

    # Last good DF (no “error” column) if any, else last error DF
    good_df  = next((d for d in reversed(dfs) if "error" not in d.columns), None)
    error_df = next((d for d in reversed(dfs) if "error" in d.columns),  None)

    # ── Display visual or error ────────────────────────────────────
    if good_df is not None:
        choose_visual(good_df)
    elif error_df is not None:
        st.error(error_df["error"][0])
    else:
        st.info("No table output to visualise.")
