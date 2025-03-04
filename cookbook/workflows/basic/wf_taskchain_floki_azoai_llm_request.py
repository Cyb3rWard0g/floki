from floki.workflow import WorkflowApp, workflow, task
from floki.types import DaprWorkflowContext
from floki import OpenAIChatClient
from dotenv import load_dotenv
import logging
import os

# Define Workflow logic
@workflow(name='lotr_workflow')
def task_chain_workflow(ctx: DaprWorkflowContext):
    result1 = yield ctx.call_activity(get_character)
    result2 = yield ctx.call_activity(get_line, input={"character": result1})
    return result2

@task(description="""
    Pick a random character from The Lord of the Rings\n
    and respond with the character's name ONLY
""")
def get_character() -> str:
    pass

@task(description="What is a famous line by {character}",)
def get_line(character: str) -> str:
    pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Load environment variables
    load_dotenv()

    # Define Azure OpenAI Client
    llm = OpenAIChatClient(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"), # or add AZURE_OPENAI_API_KEY environment variable to .env file
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), # or add AZURE_OPENAI_ENDPOINT environment variable to .env file
        azure_deployment="gpt-4o"
    )

    # Initialize the WorkflowApp
    wfapp = WorkflowApp(llm=llm)

    # Run workflow
    results = wfapp.run_and_monitor_workflow(task_chain_workflow)
    print(results)