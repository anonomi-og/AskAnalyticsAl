# agents.py
"""
SQL-aware LangChain agent that ALWAYS fetches a pandas DataFrame.

Env vars needed (see .env):
OPENAI_API_KEY
GCP_PROJECT           personal-playground-467919
BQ_DATASET            ai_bot_test
BQ_LOCATION           EU
GOOGLE_APPLICATION_CREDENTIALS
"""

from __future__ import annotations

import os
from typing import Any, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities import SQLDatabase
from langchain.tools import Tool

load_dotenv()


# ── Helpers ─────────────────────────────────────────────────────────
def make_sql_db() -> SQLDatabase:
    uri = (
        f"bigquery://{os.getenv('GCP_PROJECT')}"
        f"/{os.getenv('BQ_DATASET')}?location={os.getenv('BQ_LOCATION','EU')}"
    )
    return SQLDatabase(create_engine(uri))


def query_with_df(sql: str, db: SQLDatabase) -> "pd.DataFrame":  # noqa: F821
    """
    Execute SELECT *or aggregate* SQL against BigQuery and return a pandas DataFrame.
    """
    import pandas as pd

    if not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT statements are allowed.")
    with db.engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


# ── Build the agent ────────────────────────────────────────────────
def get_agent():
    if hasattr(get_agent, "_agent"):
        return get_agent._agent  # singleton

    db = make_sql_db()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Our single DataFrame-returning tool
    df_tool = Tool(
        name="query_with_df",
        description=(
            "Run a SQL SELECT statement on the BigQuery database and get the "
            "result as a pandas DataFrame. ALWAYS use this tool to answer any "
            "question that requires data. Example input: "
            "`SELECT ContractType, COUNT(*) AS Count "
            "FROM SampleCustomerTable GROUP BY ContractType`"
        ),
        func=lambda q: query_with_df(q, db),
        return_direct=False,
    )

    # Extra helpers so the model can discover table & column names
    schema_tool = Tool(
        name="sql_db_schema",
        description="Get the schema (DDL) for a given table name.",
        func=lambda table: db.get_table_info(table),
        return_direct=False,
    )
    tables_tool = Tool(
        name="sql_db_list_tables",
        description="List all tables in the database.",
        func=lambda _: "\n".join(db.get_usable_table_names()),
        return_direct=False,
    )

    tools = [df_tool, schema_tool, tables_tool]

    # System prompt enforces correct behaviour
    prefix = (
        "You are an expert data analyst. "
        "To answer a user question you MUST:\n"
        "1. decide the SQL you need,\n"
        "2. run it using the `query_with_df` tool,\n"
        "3. summarise or visualise the DataFrame result.\n"
        "Never rely on sample rows in the schema output."
    )
    suffix = (
        "When you have the answer, respond with a clear sentence. "
        "Do NOT output SQL unless the user explicitly asks for it."
    )

    get_agent._agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=True,
        agent_kwargs={"prefix": prefix, "suffix": suffix},
    )
    return get_agent._agent
