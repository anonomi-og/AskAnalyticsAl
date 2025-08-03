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
    with st.spinner("Thinking…"):
        agent  = get_agent()
        result = agent.invoke({"input": question})

    # — Debug: which tools did the agent call? —
    st.text(f"🛠️  Tools used: {[act.tool for act, _ in result['intermediate_steps']]}")

    # — Natural-language answer —
    st.success(result["output"])

    # — First DataFrame returned by query_with_df (if any) —
    df = next(
        (
            obs
            for act, obs in result["intermediate_steps"]
            if act.tool == "query_with_df" and isinstance(obs, pd.DataFrame)
        ),
        None,
    )

    # — Show either a chart/table or an error message —
    if df is None:
        st.info("No table output to visualise.")
    elif "error" in df.columns:
        st.error(df["error"][0])
    else:
        choose_visual(df)
