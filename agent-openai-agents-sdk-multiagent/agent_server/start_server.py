import logging
import os

from dotenv import load_dotenv
from mlflow.genai.agent_server import AgentServer, setup_mlflow_git_based_version_tracking

# Load env vars from .env before importing the agent for proper auth
load_dotenv(dotenv_path=".env", override=True)

# Need to import the agent to register the functions with the server
import agent_server.agent  # noqa: E402

logger = logging.getLogger(__name__)

agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=True)
# Define the app as a module level variable to enable multiple workers
app = agent_server.app  # noqa: F841

# Skip MLflow Databricks setup when running without Databricks (e.g. MLFLOW_TRACKING_URI=file:./mlruns)
# or when Databricks auth isn't configured yet. Set MLFLOW_SKIP_DATABRICKS_SETUP=1 to force skip.
_skip_mlflow_setup = os.environ.get("MLFLOW_SKIP_DATABRICKS_SETUP") == "1"
if not _skip_mlflow_setup:
    try:
        setup_mlflow_git_based_version_tracking()
    except Exception as exc:
        logger.warning(
            "MLflow Databricks setup failed: %s. App will run without Databricks tracing. "
            "To run without Databricks: set MLFLOW_TRACKING_URI=file:./mlruns and MLFLOW_EXPERIMENT_ID= "
            "in .env, or set MLFLOW_SKIP_DATABRICKS_SETUP=1.",
            exc,
        )


def main():
    agent_server.run(app_import_string="agent_server.start_server:app")

 