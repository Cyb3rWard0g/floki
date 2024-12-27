from floki import WorkflowApp
from floki.types import DaprWorkflowContext
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()

wfapp = WorkflowApp()

@wfapp.workflow
def question(ctx:DaprWorkflowContext, input:int):
    step1 = yield ctx.call_activity(ask, input=input)
    print(step1)
    return step1

class Dog(BaseModel):
    name: str
    bio: str
    breed: str

@wfapp.task("Who was {name}?")
def ask(name:str) -> Dog:
    pass

if __name__ == '__main__':
    wfapp.run_and_monitor_workflow(workflow=question, input="Scooby Doo")