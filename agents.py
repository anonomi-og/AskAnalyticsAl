# agents.py
from dotenv import load_dotenv
load_dotenv()

import os
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import initialize_agent, AgentType


def _make_sql_database() -> SQLDatabase:
    project   = os.getenv("GCP_PROJECT")
    dataset   = os.getenv("BQ_DATASET")
    location  = os.getenv("BQ_LOCATION", "EU")
    return SQLDatabase(create_engine(
        f"bigquery://{project}/{dataset}?location={location}"
    ))


def get_agent():
    if hasattr(get_agent, "_agent"):
        return get_agent._agent                        # ─┐ singleton

    llm      = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    sql_db   = _make_sql_database()

    # Build full toolkit, **then keep only the tools we want**
    toolkit  = SQLDatabaseToolkit(
        llm=llm,
        db=sql_db,
        include_dataframe=True,        # adds query_with_df
    )
    allowed = {"query_with_df", "sql_db_list_tables", "sql_db_schema"}
    tools   = [t for t in toolkit.get_tools() if t.name in allowed]

    get_agent._agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=True,
    )
    return get_agent._agent
