# Dapr & Floki Workflows

!!! info
    This quickstart requires `Dapr CLI` and `Docker`. You must have your [local Dapr environment set up](../installation.md).

[Dapr workflows](https://docs.dapr.io/developing-applications/building-blocks/workflow/workflow-overview/) provide a solid framework for managing long-running processes and interactions across distributed systems using the Dapr Python SDK. Floki builds on this by introducing tasks, which simplify defining and managing workflows while adding features like tool integrations and LLM-powered reasoning. This approach allows you to start with basic Dapr workflows and expand to more advanced capabilities, such as LLM-driven tasks or multi-agent coordination, as your needs grow.

## Default Dapr Workflows

Creating a [Dapr workflow](https://docs.dapr.io/developing-applications/building-blocks/workflow/workflow-overview/) is straightforward. Start by creating a python script `wf_taskchain_original_activity.py` and initializing the `WorkflowRuntime`, which manages the execution of workflows.

```python
import dapr.ext.workflow as wf

wfr = wf.WorkflowRuntime()
```

Next, define the workflow logic and the individual tasks it will execute. In this example, we pass a number through a sequence of steps, where each step performs an operation on the input. This follows the [Dapr Task chaining pattern](https://docs.dapr.io/developing-applications/building-blocks/workflow/workflow-patterns/#task-chaining) commonly used in Dapr workflows.

```python
@wfr.workflow(name='random_workflow')
def task_chain_workflow(ctx: wf.DaprWorkflowContext, wf_input: int):
    result1 = yield ctx.call_activity(step1, input=wf_input)
    result2 = yield ctx.call_activity(step2, input=result1)
    result3 = yield ctx.call_activity(step3, input=result2)
    return [result1, result2, result3]

@wfr.activity
def step1(ctx, activity_input):
    print(f'Step 1: Received input: {activity_input}.')
    # Do some work
    return activity_input + 1

@wfr.activity
def step2(ctx, activity_input):
    print(f'Step 2: Received input: {activity_input}.')
    # Do some work
    return activity_input * 2

@wfr.activity
def step3(ctx, activity_input):
    print(f'Step 3: Received input: {activity_input}.')
    # Do some work
    return activity_input ^ 2
```

Finally, start the `WorkflowRuntime`, create a `DaprWorkflowClient`, and schedule the workflow. The rest of the script monitors the workflow's progress and handles its completion.

```python
from time import sleep

if __name__ == '__main__':
    wfr.start()
    sleep(5)  # wait for workflow runtime to start

    wf_client = wf.DaprWorkflowClient()
    instance_id = wf_client.schedule_new_workflow(workflow=task_chain_workflow, input=10)
    print(f'Workflow started. Instance ID: {instance_id}')
    state = wf_client.wait_for_workflow_completion(instance_id)
    print(f'Workflow completed! Status: {state.runtime_status}')

    wfr.shutdown()
```

With this setup, you can easily implement and execute workflows using Dapr, leveraging its powerful runtime and task orchestration capabilities.

### Set Up a State Store for Workflows

Before running a workflow, you need to define a [Dapr component for the state store](https://docs.dapr.io/reference/components-reference/supported-state-stores/). Workflows in Dapr leverage the actor model under the hood, which requires a state store to manage actor states and ensure reliability. When running locally, you can use the default [Redis state store](https://docs.dapr.io/reference/components-reference/supported-state-stores/setup-redis/) that was set up during `dapr init`. To configure the state store for workflows, we add an `actorStateStore` definition, ensuring workflows have the persistence they need for fault tolerance and long-running operations.

!!! info
    Create a folder named `components` and create a `workflowstate.yaml` file inside with the following content. Name the state store `workflowstatestore`.

```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: workflowstatestore
spec:
  type: state.redis
  version: v1
  initTimeout: 1m
  metadata:
  - name: redisHost
    value: localhost:6379
  - name: redisPassword
    value: ""
  - name: actorStateStore
    value: "true"
```

### Run Workflow!

Now, you can run that workflow with the `Dapr CLI`:

```bash
dapr run --app-id originalwf --dapr-grpc-port 50001 --resources-path components/ -- python3 wf_taskchain_original_activity.py
```

![](../../img/workflows_original_activity.png)


## Dapr Workflow -> Floki Workflows

With `Floki`, the goal was to simplify workflows while adding flexibility and powerful integrations. I wanted to create a way to track the workflow state, including input, output, and status, while also streamlining monitoring. To achieve this, I built additional `workflow` and `activity` wrappers. The workflow wrapper stays mostly the same as Dapr's original, but the activity wrapper has been extended into a `task wrapper`. This change allows tasks to integrate seamlessly with LLM-based prompts and other advanced capabilities.

!!! info
    The same example as before can be written in the following way. While the difference might not be immediately noticeable, this is a straightforward example of task chaining using Python functions. Create a file named `wf_taskchain_floki_activity.py`.

```python
from floki import WorkflowApp
from floki.types import DaprWorkflowContext

wfapp = WorkflowApp()

@wfapp.workflow(name='random_workflow')
def task_chain_workflow(ctx:DaprWorkflowContext, input: int):
    result1 = yield ctx.call_activity(step1, input=input)
    result2 = yield ctx.call_activity(step2, input=result1)
    result3 = yield ctx.call_activity(step3, input=result2)
    return [result1, result2, result3]

@wfapp.task
def step1(activity_input):
    print(f'Step 1: Received input: {activity_input}.')
    # Do some work
    return activity_input + 1

@wfapp.task
def step2(activity_input):
    print(f'Step 2: Received input: {activity_input}.')
    # Do some work
    return activity_input * 2

@wfapp.task
def step3(activity_input):
    print(f'Step 3: Received input: {activity_input}.')
    # Do some work
    return activity_input ^ 2

if __name__ == '__main__':
    results = wfapp.run_and_monitor_workflow(task_chain_workflow, input=10)
    print(f"Results: {results}")
```

Now, you can run that workflow with the same command with the `Dapr CLI`:

```bash
dapr run --app-id flokiwf --dapr-grpc-port 50001 --resources-path components/ -- python3 wf_taskchain_floki_activity.py
```

![](../../img/workflows_floki_activity.png)

If we inspect the `Workflow State` in the state store, you would see something like this:

```json
{
    "instances":{
        "a0d2de00818e4f0098318f0bed5fa1ee":{
            "input":"10",
            "output":"[11, 22, 20]",
            "status":"completed",
            "start_time":"2024-11-27T16:57:14.465235",
            "end_time":"2024-11-27T16:57:17.540588",
            "messages":[]
        }
    }
}
```

`Floki` processes the workflow execution and even extracts the final output.
