# LLM-based Task Workflows

!!! info
    This quickstart requires `Dapr CLI` and `Docker`. You must have your [local Dapr environment set up](../installation.md).

In `Floki`, LLM-based Task Workflows allow developers to design step-by-step workflows where LLMs provide reasoning and decision-making at defined stages. These workflows are deterministic and structured, enabling the execution of tasks in a specific order, often defined by Python functions. This approach does not rely on event-driven systems or pub/sub messaging but focuses on defining and orchestrating tasks with the help of LLM reasoning when necessary. Ideal for scenarios that require a predefined flow of tasks enhanced by language model insights.

Now that we have a better understanding of `Dapr` and `Floki` workflows, let’s explore how to use Dapr activities or Floki tasks to call LLM Inference APIs, such as [OpenAI Tex Generation endpoint](https://platform.openai.com/docs/guides/text-generation), with models like `gpt-4o`.

## Dapr Workflows & LLM Inference APIs

To start, we can define a few `Dapr` activities that interact with `OpenAI APIs`. These activities can be chained together so the output of one step becomes the input for the next. For example, in the first step, we can use the LLM’s parameteric knowledge to pick a random character from [The Lord of the Rings](https://en.wikipedia.org/wiki/The_Lord_of_the_Rings). In the second step, the LLM can generate a famous line spoken by that character.

!!! tip
    Make sure you have your environment variables set up in an `.env` file so that the library can pick it up and use it to communicate with `OpenAI` services. We set them up in the [LLM Inference Client](llm.md) section

Start by initializing the `WorkflowRuntime` and gathering the right environment variables.

```python
import dapr.ext.workflow as wf
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize Workflow Instance
wfr = wf.WorkflowRuntime()
```

Next, let's define a workflow and activities, demonstrating how each step integrates with the OpenAI Python SDK to achieve this functionality.

```python
# Define Workflow logic
@wfr.workflow(name='lotr_workflow')
def task_chain_workflow(ctx: wf.DaprWorkflowContext):
    result1 = yield ctx.call_activity(get_character)
    result2 = yield ctx.call_activity(get_line, input=result1)
    return result2

# Activity 1
@wfr.activity(name='step1')
def get_character(ctx):
    client = OpenAI()
    response = client.chat.completions.create(
        messages = [
            {
                "role": "user",
                "content": "Pick a random character from The Lord of the Rings and respond with the character name only"
            }
        ],
        model = 'gpt-4o'
    )
    character = response.choices[0].message.content
    print(f"Character: {character}")
    return character

# Activity 2
@wfr.activity(name='step2')
def get_line(ctx, character: str):
    client = OpenAI()
    response = client.chat.completions.create(
        messages = [
            {
                "role": "user",
                "content": f"What is a famous line by {character}"
            }
        ],
        model = 'gpt-4o'
    )
    line = response.choices[0].message.content
    print(f"Line: {line}")
    return line
```

Finally, we complete the process by triggering the workflow and handling the output, including waiting for the workflow to finish before processing the results.

```python
from time import sleep

if __name__ == '__main__':
    wfr.start()
    sleep(5)  # wait for workflow runtime to start

    wf_client = wf.DaprWorkflowClient()
    instance_id = wf_client.schedule_new_workflow(workflow=task_chain_workflow)
    print(f'Workflow started. Instance ID: {instance_id}')
    state = wf_client.wait_for_workflow_completion(instance_id)
    print(f'Workflow completed! Status: {state.runtime_status}')

    wfr.shutdown()
```

!!! tip
    Before running a workflow, remember that you need to define a [Dapr component for the state store](https://docs.dapr.io/reference/components-reference/supported-state-stores/).

```bash
dapr run --app-id originalllmwf --dapr-grpc-port 50001 --resources-path components/ -- python3 wf_taskchain_openai_original_llm_request.py
```

![](../../img/workflows_originial_llm_request.png)

## Floki LLM-based Tasks

Now, let’s get to the exciting part! `Tasks` in `Floki` build on the concept of `activities` and bring additional flexibility. Using Python function signatures, you can define tasks with ease. The `task decorator` allows you to provide a `description` parameter, which acts as a prompt for the default LLM inference client in `Floki` (`OpenAIChatClient` by default).

You can also use function arguments to pass variables to the prompt, letting you dynamically format the prompt before it’s sent to the text generation endpoint. This makes it simple to implement workflows that follow the [Dapr Task chaining pattern](https://docs.dapr.io/developing-applications/building-blocks/workflow/workflow-patterns/#task-chaining), just like in the earlier example, but with even more flexibility.

```python
from floki import WorkflowApp
from floki.types import DaprWorkflowContext
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the WorkflowApp
wfapp = WorkflowApp()

# Define Workflow logic
@wfapp.workflow(name='lotr_workflow')
def task_chain_workflow(ctx: DaprWorkflowContext):
    result1 = yield ctx.call_activity(get_character)
    result2 = yield ctx.call_activity(get_line, input={"character": result1})
    return result2

@wfapp.task(description="""
    Pick a random character from The Lord of the Rings\n
    and respond with the character's name only
""")
def get_character() -> str:
    pass

@wfapp.task(description="What is a famous line by {character}",)
def get_line(character: str) -> str:
    pass

if __name__ == '__main__':
    results = wfapp.run_and_monitor_workflow(task_chain_workflow)
    print(f"Famous Line: {results}")
```

Run the workflow with the following command:

```bash
dapr run --app-id flokillmmwf --dapr-grpc-port 50001 --resources-path components/ -- python3 wf_taskchain_openai_floki_llm_request.py
```

![](../../img/workflows_floki_llm_request.png)