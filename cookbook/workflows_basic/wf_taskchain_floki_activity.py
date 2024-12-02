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