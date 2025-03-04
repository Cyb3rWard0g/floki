from floki.workflow import WorkflowApp, workflow, task
from floki.types import DaprWorkflowContext
from pydantic import BaseModel
from floki import OpenAIChatClient
from dotenv import load_dotenv
import logging
import os

@workflow
def question(ctx:DaprWorkflowContext, input:int):
    step1 = yield ctx.call_activity(ask, input=input)
    return step1

class Dog(BaseModel):
    name: str
    bio: str
    breed: str

@task("Who was {name}?")
def ask(name:str) -> Dog:
    pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Load environment variables
    load_dotenv()

    # Define Azure OpenAI Client
    llm = OpenAIChatClient(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"), # or add AZURE_OPENAI_API_KEY environment variable to .env file
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), # or add AZURE_OPENAI_ENDPOINT environment variable to .env file
        azure_deployment="gpt-4o",
        api_version="2024-08-01-preview" # response_format value as json_schema is enabled only for api versions 2024-08-01-preview and later
    )

    # Initialize the WorkflowApp
    wfapp = WorkflowApp(llm=llm)

    # Run Workflow
    results = wfapp.run_and_monitor_workflow(workflow=question, input="Scooby Doo")
    print(results)