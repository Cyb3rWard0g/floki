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

NEXT_STEP_PROMPT = """## Task Context

The team is working on the following task:

{task}

### Team of Agents (ONLY these agents are available):
{agents}

### Current Execution Plan:
{plan}

### Next Steps:
- **Select the next best-suited agent** from the team of agents list who should respond **based on the execution plan above**.
- **DO NOT select an agent that is not explicitly listed in `{agents}`**.
- **You must always provide a valid agent name** from the team**. DO NOT return `null` or an empty agent name**.
- Provide a **clear, actionable instruction** for the next agent.
- **You must ONLY select step and substep IDs that EXIST in the plan.** 
  - **A substep should ONLY be selected if it is currently `"not_started"` or `"in_progress"`**.
  - **If the main step is `"not_started"` but has `"completed"` substeps, you must correctly identify the next `"not_started"` substep.**
  - **DO NOT create or assume non-existent step/substep IDs.**
  - **DO NOT reference any invalid identifiers. Always check the plan.**

### Expected Output Format (JSON Schema):
{next_step_schema}
"""

PROGRESS_CHECK_PROMPT = """## Progress Check

### Task Context
The team is working on the following task:

{task}

### Current Execution Plan:

{plan}

### Latest Execution Context:
- **Step ID:** {step}  
- **Substep ID (if applicable):** {substep}  
- **Step Execution Results:** "{results}"  

### Task Evaluation:
Assess the task progress based on **conversation history**, execution results, and the structured plan.

1. **Determine Overall Task Verdict**  
   - `"continue"` → The task is still in progress and requires further work.  
   - `"completed"` → The task has been successfully completed (i.e., all steps are marked `"completed"`).  
   - `"failed"` → The task cannot be completed due to an unresolved issue.  

2. **Evaluate Step Completion**  
   - If the **executed step or sub-step (above) is explicitly completed**, mark it as `"completed"`.  
   - If the executed step **has sub-steps**, ensure **all** sub-steps are `"completed"` before marking the parent step `"completed"`.  

3. **Update Step & Sub-Step Status**  
   - **Only mark a step as `"completed"` if it has been explicitly confirmed as done.**  
   - **Do NOT transition `"not_started"` steps to `"in_progress"` here**. That is handled in a different process.  
   - **Only update statuses if the verdict is `"continue"`**.  

4. **Plan Adjustments (Only If Necessary)**  
   - If the step descriptions are **unclear or incomplete**, update `"plan_restructure"` with a **single modified step**.  
   - Do **not** introduce unnecessary modifications.  

### Important:
- **Do NOT mark a step as `"completed"` unless explicitly confirmed based on execution results.**  
- **Do NOT transition `"not_started"` steps to `"in_progress"` here**. This happens in a different process.  
- **Only update statuses and restructure steps when necessary.**  
- **Do not make unnecessary modifications to the plan.**  

### Expected Output Format (JSON Schema):
{progress_check_schema}
"""

SUMMARY_GENERATION_PROMPT = """# Summary Generator

## Initial Task:
{task}

## Execution Overview:
- **Final Verdict:** {verdict}  
  _(Possible values: `"continue"`, `"completed"`, `"failed"` `max_iterations_reached`)_  
- **Execution Plan Status:**  
  {plan}  
- **Last Action Taken:**
  - **Step:** `{step}` (Sub-step `{substep}` if applicable)
  - **Executing Agent:** `{agent}`
  - **Execution Result:** {result}

## Instructions to Generate Best Summary
Based on the **conversation history** and **execution plan**, generate a **clear and structured** summary:

1. **If the task is `"completed"`**, provide a concise but complete final summary.
   - **Briefly describe the key steps taken** and the final outcome.
   - Ensure clarity, avoiding excessive details while **focusing on essential takeaways**.
   - Phrase it **as if reporting back to the user** rather than as a system log.

2. **If the task is `"failed"`**, explain why.
   - Summarize blockers, unresolved challenges, or missing steps.
   - If applicable, suggest potential next actions to resolve the issue.

3. **If the task is `"continue"`**, summarize progress so far.
   - **Highlight completed steps** and the current state of execution.
   - **Mention what remains unfinished** and the next logical step.
   - Keep it informative but **concise and forward-looking**.

4. **If the task is `"max_iterations_reached"`**, summarize the progress and note the limitation.
   - Highlight what has been **achieved so far** and what **remains unfinished**.
   - Clearly state that the **workflow reached its iteration limit** before completion.
   - If possible, **suggest next steps** (e.g., refining the execution plan, restarting the task, or adjusting agent strategies).

## Expected Output
A structured summary that is:
- **Clear and to the point**  
- **Context-aware (includes results and execution progress)**  
- **User-friendly (reads naturally rather than like system logs)**  
- **Relevant (avoids unnecessary details while maintaining accuracy)** 
"""