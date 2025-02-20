TASK_PLANNING_PROMPT = """## Task Overview
We need to develop a structured, **logically ordered** plan to accomplish the following task:

{task}

### Team of Agents
{agents}

### Planning Instructions:
1. **Break the task into a structured, step-by-step plan** ensuring logical sequencing.
2. **Each step should be described in a neutral, objective manner**, focusing on the action required rather than specifying who performs it.
3. **The plan should be designed to be executed by the provided team**, but **DO NOT reference specific agents** in step descriptions.
4. **Do not introduce additional roles, skills, or external resources** outside of the given team.
5. **Use clear and precise descriptions** for each step, focusing on **what** needs to be done rather than **who** does it.
6. **Avoid unnecessary steps or excessive granularity**. Keep the breakdown clear and efficient.
7. **Maintain a natural task flow** where each step logically follows the previous one.
8. **Each step must have a `status` field** to track progress (Initially all set to `not_started`).
9. **Sub-steps should be included only if a step requires multiple distinct actions.**
10. **Focus only on structuring the task execution**, NOT on assigning agents at this stage.

### Expected Output Format (JSON Schema):
{plan_schema}
"""

TASK_INITIAL_PROMPT = """## Mission Briefing

We have received the following task:

{task}

### Team of Agents
{agents}

### Execution Plan
Here is the structured approach the team will follow to accomplish the task:

{plan}
"""

PROGRESS_CHECK_PROMPT = """## Progress Check

### Task Context
The team is working on the following task:

{task}

### Current Execution Plan:

{plan}

### Status Update:
Review the current plan and progress to update the following:
1. If this is the **first iteration**, mark **only** the first step (or its first sub-step, if applicable) as `"in_progress"` instead of `"completed"` prematurely.
2. If any step or sub-step is **completed**, mark its status as `"completed"` and transition the next logical step or sub-step to `"in_progress"` based on conversation history.  
3. If no updates are needed, set `"plan_needs_update": false"`.  
4. If restructuring is required (e.g., a step needs to be rewritten or sub-steps need to be added), update `"plan_restructure"` with **only the modified step**. 
5. **DO NOT** add unnecessary changes.

### Important:
- **Ensure steps progress logically—steps should not be marked `"completed"` unless they have been explicitly confirmed as done based on the conversation history.**
- **If a step has sub-steps, track sub-step progress first before transitioning the main step to `"completed"`.**
- **Only update statuses and restructure steps when necessary.**
- **Do not make unnecessary modifications to the plan.**

### Expected Output Format (JSON Schema):
{progress_check_schema}
"""

NEXT_STEP_PROMPT = """## Task Context

The team is working on the following task:

{task}

### Team of Agents (ONLY these agents are available):
{agents}

### Current Execution Plan:
{plan}

### Current Progress:
Review the plan and determine the overall task status:
1. **"continue"** - The task is still in progress and requires further work.
2. **"completed"** - The task has been successfully completed (i.e., all steps are marked `"completed"`).
3. **"failed"** - The task cannot be completed due to an unresolved issue.

### Next Steps:
- If `"continue"`, **select the next best-suited agent from the provided list** who should respond **based on the execution plan**.
- **DO NOT select an agent that is not explicitly listed in `{agents}`**.
- **You must always provide a valid agent name** from the team—**DO NOT return `null` or an empty agent name**.
- Provide a **clear, actionable instruction** for the next agent.
- **Indicate the `step` and `substep` ids (if applicable) that this agent will be working on.**

### Expected Output Format (JSON Schema):
{next_step_schema}
"""