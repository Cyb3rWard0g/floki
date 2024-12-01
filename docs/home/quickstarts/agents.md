# LLM-based AI Agents

In the `Floki` framework, agents are autonomous systems powered by large language models (LLMs) that serve as their reasoning engine. These agents use the LLM’s parametric knowledge to process information, reason in natural language, and interact dynamically with their environment by leveraging tools. Tools allow the agents to perform real-world tasks, gather new information, and adapt their reasoning based on feedback.

!!! info
    By default, `Floki` sets the agentic pattern for the `Agent` class to `toolcalling` mode, enabling AI agents to interact dynamically with external tools using [OpenAI's Function Calling](https://platform.openai.com/docs/guides/function-calling?ref=blog.openthreatresearch.com).

`Tool Calling` empowers agents to identify the right tools for a task, format the necessary arguments, and execute the tools independently. The results are then passed back to the LLM for further processing, enabling seamless and adaptive agent workflows.

## Environment Variables

Create an `.env` file for your API keys and other environment variables with sensitive information that you do not want to hardcode.

```
OPENAI_API_KEY="XXXXXX"
OPENAI_BASE_URL="https://api.openai.com/v1"
```

Use [Python-dotenv](https://pypi.org/project/python-dotenv/) to load environment variables from `.env`.

```
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
```

## Create a Basic Agent

In `Floki`, tools bridge basic Python functions and `OpenAI's Function Calling` format, enabling seamless interaction between agents and external tasks. You can use `Pydantic` models to define the schema for tool arguments, ensuring structured input and validation.

By annotating functions with `@tool` and specifying the argument schema, you transform them into `Agent tools` that can be invoke dynamically during workflows. This approach makes your tools compatible with LLM-driven decision-making and execution.

```python
from floki import tool
from pydantic import BaseModel, Field

class GetWeatherSchema(BaseModel):
    location: str = Field(description="location to get weather for")

@tool(args_model=GetWeatherSchema)
def get_weather(location: str) -> str:
    """Get weather information based on location."""
    import random
    temperature = random.randint(60, 80)
    return f"{location}: {temperature}F."

class JumpSchema(BaseModel):
    distance: str = Field(description="Distance for agent to jump")

@tool(args_model=JumpSchema)
def jump(distance: str) -> str:
    """Jump a specific distance."""
    return f"I jumped the following distance {distance}"

tools = [get_weather,jump]
```

Next, create your Agent by specifying key attributes such as `name`, `role`, `goal`, and `instructions`, while assigning the `tools` defined earlier. This setup equips your agent with a clear purpose and the ability to interact dynamically with its environment.

```python
from floki import Agent

AIAgent = Agent(
    name = "Stevie",
    role = "Weather Assistant",
    goal = "Assist Humans with weather related tasks.",
    instructions = ["Get accurate weather information", "From time to time, you can Jump."],
    tools=tools
)
```

Finally, run the agent with a task.

```python
AIAgent.run("What is the weather in Virgina, New York and Washington DC?")
```

```
user:
What is the weather in Virgina, New York and Washington DC?
------------------------------------------------------------------------------
assistant(tool_call):
Function name: GetWeather (Call Id: call_QOxUTdkWXhA5hlaKfEmvY3As)
Arguments: {"location": "Virginia"}
--------------------------------------------------------------------------------
assistant(tool_call):
Function name: GetWeather (Call Id: call_brouCb5MnCgPbK172dKaw0cx)
Arguments: {"location": "New York"}
--------------------------------------------------------------------------------
assistant(tool_call):
Function name: GetWeather (Call Id: call_KUH1ErAHdMFV83gYYghRdaIK)
Arguments: {"location": "Washington DC"}
-------------------------------------------------------------------------------
tool(Id: call_QOxUTdkWXhA5hlaKfEmvY3As):
Virginia: 68F.
-------------------------------------------------------------------------------
tool(Id: call_brouCb5MnCgPbK172dKaw0cx):
New York: 62F.
--------------------------------------------------------------------------------
tool(Id: call_KUH1ErAHdMFV83gYYghRdaIK):
Washington DC: 67F.
--------------------------------------------------------------------------------
assistant:
Here is the current weather for each location:
- **Virginia**: 68°F
- **New York**: 62°F
- **Washington DC**: 67°F
-------------------------------------------------------------------------------
```

You can check the agent's `chat_history` property.

```python
AIAgent.chat_history
```

```
[{'role': 'user',
  'content': 'What is the weather in Virgina, New York and Washington DC?'},
 {'content': 'Here is the current weather for each location:\n\n- **Virginia**: 68°F\n- **New York**: 62°F\n- **Washington DC**: 67°F',
  'role': 'assistant'}]
```

You can also reset the agent's memory:

```python
AIAgent.reset_memory()
```