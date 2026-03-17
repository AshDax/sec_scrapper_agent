import asyncio
import logging

import mlflow
from dotenv import load_dotenv
from mlflow.genai.agent_server import get_invoke_function
from mlflow.genai.scorers import (
    Completeness,
    ConversationalSafety,
    ConversationCompleteness,
    Fluency,
    KnowledgeRetention,
    RelevanceToQuery,
    Safety,
    ToolCallCorrectness,
    UserFrustration,
)
from mlflow.genai.simulators import ConversationSimulator
from mlflow.types.responses import ResponsesAgentRequest

load_dotenv(dotenv_path=".env", override=True)
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)

from agent_server import agent  # noqa: F401, E402

test_cases = [
    {
        "goal": "Extract business attributes from a sample SEC 10-K filing",
        "persona": "A data analyst who needs to enrich business records with SEC filing data.",
        "simulation_guidelines": [
            "Provide a sample SEC filing text and ask the agent to extract all business attributes.",
            "Ask follow-up questions about specific attributes like employee count or revenue.",
        ],
    },
    {
        "goal": "Determine if a company is a manufacturer based on its SEC filing description",
        "persona": "A business researcher who needs to classify companies by industry type.",
        "simulation_guidelines": [
            "Ask the agent to analyze a company description and determine if it is a manufacturer.",
            "Ask for the evidence and reasoning behind the classification.",
        ],
    },
    {
        "goal": "Find the headquarters address and contact information for a company from SEC filings",
        "persona": "A sales operations analyst building a contact database.",
        "simulation_guidelines": [
            "Ask the agent to find the company's address, phone number, and website.",
            "Verify the extracted information is in a structured format.",
        ],
    },
]

simulator = ConversationSimulator(
    test_cases=test_cases,
    max_turns=4,
    user_model="databricks:/databricks-claude-sonnet-4-5",
)

invoke_fn = get_invoke_function()
assert invoke_fn is not None, (
    "No function registered with the `@invoke` decorator found. "
    "Ensure you have a function decorated with `@invoke()`."
)

if asyncio.iscoroutinefunction(invoke_fn):
    import nest_asyncio

    nest_asyncio.apply()

    def predict_fn(input: list[dict], **kwargs) -> dict:
        req = ResponsesAgentRequest(input=input)
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(invoke_fn(req))
        return response.model_dump()
else:

    def predict_fn(input: list[dict], **kwargs) -> dict:
        req = ResponsesAgentRequest(input=input)
        response = invoke_fn(req)
        return response.model_dump()


def evaluate():
    mlflow.genai.evaluate(
        data=simulator,
        predict_fn=predict_fn,
        scorers=[
            Completeness(),
            ConversationCompleteness(),
            ConversationalSafety(),
            KnowledgeRetention(),
            UserFrustration(),
            Fluency(),
            RelevanceToQuery(),
            Safety(),
            ToolCallCorrectness(),
        ],
    )
