# agents.py
"""
Creates a LangChain SQL-aware agent that speaks to BigQuery.
Reads config from environment variables set in .env
"""

# agents.py   (only the imports + get_agent() change)
from dotenv import load_dotenv
load_dotenv()

import os
from sqlalchemy import create_engine

# NEW imports
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI


def _make_sql_database() -> SQLDatabase:
    project   = os.getenv("GCP_PROJECT")
    dataset   = os.getenv("BQ_DATASET")
    location  = os.getenv("BQ_LOCATION", "EU")
    uri = f"bigquery://{project}/{dataset}?location={location}"
    return SQLDatabase(create_engine(uri))


def get_agent():
    """Singleton SQL agent that can return BOTH text answers AND a DataFrame."""
    if not hasattr(get_agent, "_agent"):
        llm      = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        sql_db   = _make_sql_database()
        toolkit  = SQLDatabaseToolkit(llm=llm, db=sql_db, include_dataframe=True)

        get_agent._agent = initialize_agent(
            toolkit.get_tools(),          # includes query_with_df
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            return_intermediate_steps=True,
        )
    return get_agent._agent
