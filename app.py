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
        agent   = get_agent()
        result  = agent(question)

    st.success(result["output"])

    # ── NEW: grab the first pandas DataFrame in intermediate steps ──
    df = next(
        (
            obs
            for act, obs in result["intermediate_steps"]
            if act.tool == "query_with_df" and isinstance(obs, pd.DataFrame)
        ),
        None,
    )
    choose_visual(df)
