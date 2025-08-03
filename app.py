# app.py   (changes marked ✱)
import streamlit as st
from agents import get_agent
from viz import choose_visual
import pandas as pd

st.set_page_config(page_title="AI Data Assistant", layout="wide")
st.title("Ask your customer data")

question = st.text_input(
    "Natural-language question",
    placeholder="e.g. Show a breakdown of contract types",
)

if st.button("Run", type="primary") and question:
    with st.spinner("Thinking…"):
        agent   = get_agent()
        result  = agent(question)        # ✱ call the agent as a function
                                          #   to keep intermediate steps

    # ── Extract answer text ───────────────────────────────────────────
    answer_text = result["output"]

    # ── Extract DataFrame (if any) ────────────────────────────────────
    df = None
    for act, obs in result["intermediate_steps"]:
        if act.tool == "query_with_df":   # ✱ our DataFrame tool
            df = obs                      # obs *is* the pandas DataFrame
            break

    # ── Display ───────────────────────────────────────────────────────
    st.success(answer_text)
    choose_visual(df)
