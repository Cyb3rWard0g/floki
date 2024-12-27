# Multi-Agent Event-Driven Workflows

!!! info
    This quickstart requires `Dapr CLI` and `Docker`. You must have your [local Dapr environment set up](../installation.md).

Event-Driven Agentic Workflows in `Floki` take advantage of an event-driven system using pub/sub messaging and a shared message bus. Agents operate as autonomous entities that respond to events dynamically, enabling real-time interactions and collaboration. These workflows are highly adaptable, allowing agents to communicate, share tasks, and reason through events triggered by their environment. This approach is best suited for decentralized systems requiring dynamic agent collaboration across distributed applications.

!!! tip
    We will demonstrate this concept using the [Multi-Agent Workflow Guide](https://github.com/Cyb3rWard0g/floki/tree/main/cookbook/workflows/multi_agent_lotr) from our Cookbook, which outlines a step-by-step guide to implementing a basic agentic workflow.

## Agents as Services

In `Floki`, agents can be exposed as services, making them reusable, modular, and easy to integrate into event-driven workflows. Each agent runs as a microservice, wrapped in a [Dapr-enabled FastAPI server](https://docs.dapr.io/developing-applications/sdks/python/python-sdk-extensions/python-fastapi/). This design allows agents to operate independently while communicating through [Dapr’s pub/sub](https://docs.dapr.io/developing-applications/building-blocks/pubsub/pubsub-overview/) messaging and interacting with state stores or other services.

The way to structure such a project is straightforward. We organize our services into a directory that contains individual folders for each agent, along with a components/ directory for Dapr configurations. Each agent service includes its own app.py file, where the FastAPI server and the agent logic are defined.

```
components/                # Dapr configuration files
├── statestore.yaml        # State store configuration
├── pubsub.yaml            # Pub/Sub configuration
└── ...                    # Other Dapr components
services/                  # Directory for agent services
├── agent1/                # First agent's service
│   ├── app.py             # FastAPI app for agent1
│   └── ...                # Additional agent1 files
│── agent2/                # Second agent's service
│   ├── app.py             # FastAPI app for agent2
│   └── ...                # Additional agent2 files
└── ...                    # More agents
```

## Your First Service

Let's start by definining a `Hobbit` service with a specific `name`, `role`, `goal` and `instructions`.

```
services/                  # Directory for agent services
├── hobbit/                # Hobbit Service
│   ├── app.py             # Dapr Enabled FastAPI app for Hobbit
```

Create the `app.py` script and provide the following information.

```python
from floki import Agent, AgentService
from dotenv import load_dotenv
import asyncio
import logging

async def main():
    try:
        # Define Agent
        hobbit_agent = Agent(
            role="Hobbit",
            name="Frodo",
            goal="Take the ring to Mordor",
            instructions=["Speak like Frodo"]
        )
        # Expose Agent as a Service
        hobbit_service = AgentService(
            agent=hobbit_agent,
            message_bus_name="messagepubsub",
            agents_state_store_name="agentstatestore",
            port=8001,
            daprGrpcPort=50001
        )
        await hobbit_service.start()
    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
```

Now, you can define multiple services following this format, but it's essential to pay attention to key areas to ensure everything runs smoothly. Specifically, focus on correctly configuring the components (e.g., `statestore` and `pubsub` names) and incrementing the ports for each service.

Key Considerations:

* Ensure the `message_bus_name` matches the `pub/sub` component name in your `pubsub.yaml` file.
* Verify the `agents_state_store_name` matches the state store component defined in your `statestore.yaml` file.
* Increment the port for each new agent service (e.g., 8001, 8002, 8003).
* Similarly, increment the `daprGrpcPort` for each service (e.g., 50001, 50002, 50003) to avoid conflicts.
* Customize the Agent parameters (`role`, `name`, `goal`, and `instructions`) to match the behavior you want for each service.

## The Agentic Workflow Service

The Agentic Workflow Service in Floki extends workflows to orchestrate communication among agents. It allows you to send messages to agents to trigger their participation and monitors a shared message bus to listen for all messages being passed. This enables dynamic collaboration and task distribution among agents.

Types of Agentic Workflows:

* **Random**: Distributes tasks to agents randomly, ensuring a non-deterministic selection of participating agents for each task.
* **RoundRobin**: Cycles through agents in a fixed order, ensuring each agent has an equal opportunity to participate in tasks.
* **LLM-based**: Leverages an LLM to decide which agent to trigger based on the content and context of the task and chat history.

Next, we’ll define a `RoundRobin Agentic Workflow Service` to demonstrate how this concept can be implemented.

```python
from floki import RoundRobinWorkflowService
from dotenv import load_dotenv
import asyncio
import logging

async def main():
    try:
        roundrobin_workflow_service = RoundRobinWorkflowService(
            name="Orchestrator",
            message_bus_name="messagepubsub",
            agents_state_store_name="agentstatestore",
            workflow_state_store_name="workflowstatestore",
            port=8004,
            daprGrpcPort=50004,
            max_iterations=2
        )

        await roundrobin_workflow_service.start()
    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(main())
```

Unlike `Agents as Services`, the `Agentic Workflow Service` does not require an agent parameter since it orchestrates communication among multiple agents rather than representing a single agent. Instead, the configuration focuses on workflow-specific parameters:

* **Max Iterations**: Defines the maximum number of iterations the workflow will perform, ensuring controlled task execution and preventing infinite loops.
* **Workflow State Store Name**: Specifies the state store used to persist the workflow’s state, allowing for reliable recovery and tracking of workflow progress.
* **LLM Inference Client**: Although an individual agent is not required, the LLM-based Agentic Workflow Service depends on an LLM Inference Client. By default, it uses the [OpenAIChatClient()](https://github.com/Cyb3rWard0g/floki/blob/main/src/floki/llm/openai/chat.py) from the Floki library.

These differences reflect the distinct purpose of the Agentic Workflow Service, which acts as a centralized orchestrator rather than an individual agent service. The inclusion of the LLM Inference Client in the LLM-based workflows allows the orchestrator to leverage natural language processing for intelligent task routing and decision-making.

## The Multi-App Run template file

The Multi-App Run Template File is a YAML configuration file named `dapr.yaml` that allows you to run multiple applications simultaneously. This file is placed at the same level as the `components/` and `services/` directories, ensuring a consistent and organized project structure.

```
dapr.yaml                  # The Multi-App Run template
components/                # Dapr configuration files
├── statestore.yaml        # State store configuration
├── pubsub.yaml            # Pub/Sub configuration
└── ...                    # Other Dapr components
services/                  # Directory for agent services
├── agent1/                # First agent's service
│   ├── app.py             # FastAPI app for agent1
│   └── ...                # Additional agent1 files
│── agent2/                # Second agent's service
│   ├── app.py             # FastAPI app for agent2
│   └── ...                # Additional agent2 files
└── ...                    # More agents
```

Following our current scenario, we can set the following `Multi-App Run` template file:

```yaml
# https://docs.dapr.io/developing-applications/local-development/multi-app-dapr-run/multi-app-template/#template-properties
version: 1
common:
  resourcesPath: ./components
  logLevel: info
  appLogDestination: console
  daprdLogDestination: console

apps:
- appId: HobbitApp
  appDirPath: ./services/hobbit/
  appPort: 8001
  command: ["python3", "app.py"]
  daprGRPCPort: 50001

- appId: WizardApp
  appDirPath: ./services/wizard/
  appPort: 8002
  command: ["python3", "app.py"]
  daprGRPCPort: 50002

- appId: ElfApp
  appDirPath: ./services/elf/
  appPort: 8003
  command: ["python3", "app.py"]
  daprGRPCPort: 50003

- appId: WorkflowApp
  appDirPath: ./services/workflow-llm/
  appPort: 8004
  command: ["python3", "app.py"]
  daprGRPCPort: 50004
```

## Starting All Service Servers

!!! tip
    Make sure you have your environment variables set up in an `.env` file so that the library can pick it up and use it to communicate with `OpenAI` services. We set them up in the [LLM Inference Client](llm.md) section

To start all the service servers defined in your project, you can use the Dapr CLI with the Multi-App Run template file. When you provide a directory path, the CLI will look for the dapr.yaml file (the default name for the template) in that directory. If the file is not found, the CLI will return an error.

To execute the command, ensure you are in the root directory where the dapr.yaml file is located, then run:

```bash
dapr run -f .
```

This command reads the `dapr.yaml` file and starts all the services specified in the template.

## Monitor Services Initialization

- Verify console Logs: Each service outputs logs to confirm successful initialization.

![](../../img/workflows_roundrobin_agent_initialization.png)

- Verify Redis entries: Access the Redis Insight interface at `http://localhost:5540/`

![](../../img/workflows_roundrobin_redis_agents_metadata.png)

- Verify your agents are healthy: Check the console logs. You should see the following:

![](../../img/workflows_roundrobin_agents_health.png)

## Start Workflow via an HTTP Request

Once all services are running, you can initiate the workflow by making an HTTP POST request to the Agentic Workflow Service. This service orchestrates the workflow, triggering agent actions and handling communication among agents.

Here’s an example of how to start the workflow using `curl`:

```bash
curl -i -X POST http://localhost:8004/RunWorkflow \
    -H "Content-Type: application/json" \
    -d '{"message": "How to get to Mordor? Lets all help!"}'
```

```
HTTP/1.1 200 OK
date: Thu, 05 Dec 2024 07:46:19 GMT
server: uvicorn
content-length: 104
content-type: application/json

{"message":"Workflow initiated successfully.","workflow_instance_id":"422ab3c3f58f4221a36b36c05fefb99b"}
```

In this example:

* The request is sent to the Agentic Workflow Service running on port 8004.
* The message parameter is passed as input to the workflow, which the agents will process.
* This command demonstrates how to interact with the Agentic Workflow Service to kick off a new workflow.

## Monitoring Workflow Execution

- Check console logs to trace activities in the workflow.

![](../../img/workflows_roundrobin_console_logs_activities.png)

- Verify Redis entries: Access the Redis Insight interface at `http://localhost:5540/`

![](../../img/workflows_roundrobin_redis_broadcast_channel.png)

- As mentioned earlier, when we ran dapr init, Dapr initialized, a `Zipkin` container instance, used for observability and tracing. Open `http://localhost:9411/zipkin/` in your browser to view traces > Find a Trace > Run Query.

![](../../img/workflows_roundrobin_zipkin_portal.png)

- Select the trace entry with multiple spans labeled `<workflow name>: /taskhubsidecarservice/startinstance.`. When you open this entry, you’ll see details about how each task or activity in the workflow was executed. If any task failed, the error will also be visible here.

![](../../img/workflows_roundrobin_zipkin_spans.png)

- Check console logs to validate if workflow was executed successfuly.

![](../../img/workflows_roundrobin_console_logs_complete.png)

## Customizing the Workflow

The default setup uses the [workflow-roundrobin service](https://github.com/Cyb3rWard0g/floki/blob/main/cookbook/workflows/multi_agent_lotr/services/workflow-roundrobin/app.py), which processes agent tasks in a `round-robin` order. However, you can easily switch to a different workflow type by updating the `dapr.yaml` file.

### Available Workflow Options

* **RoundRobin**: Cycles through agents in a fixed order, ensuring each agent gets an equal opportunity to process tasks.
* **Random**: Selects an agent randomly for each task.
* **LLM-based**: Uses a large language model (e.g., GPT-4o) to determine the most suitable agent based on the message and context.

### Switching to the LLM-based Workflow

- Set Up Environment Variables: Create an `.env` file to securely store your API keys and other sensitive information. For example:

```
OPENAI_API_KEY="your-api-key"
OPENAI_BASE_URL="https://api.openai.com/v1"
```

- Update dapr.yaml: Modify the appDirPath for the workflow service to point to the workflow-llm directory:

```yaml
- appId: WorkflowApp
  appDirPath: ./services/workflow-llm/
  appPort: 8004
  command: ["python3", "app.py"]
  daprGRPCPort: 50004
```

- Load Environment Variables: Ensure your service script uses Python-dotenv to load these variables automatically:

```python
from dotenv import load_dotenv
load_dotenv()  # Load variables from .env
```

With these updates, the workflow will use the `LLM` to intelligently decide which agent to activate.

### Reset Redis Database

1. Access the Redis Insight interface at `http://localhost:5540/`
2. In the search bar type `*` to select all items in the database.
3. Click on `Bulk Actions` > `Delete` > `Delete`

![](../../img/workflows_roundrobin_redis_reset.png)

You should see an empty database now:

![](../../img/workflows_roundrobin_redis_empty.png)

### Testing the LLM-based Workflow

Restart the services with `dapr run -f` . and send a message to the workflow. Ensure your `.env` file is configured correctly and contains the necessary credentials.
