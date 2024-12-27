# Multi-Agent LOTR: Event-Driven Workflow

This guide shows you how to set up and run an event-driven agentic workflow using Floki. By leveraging [Dapr Pub/Sub](https://docs.dapr.io/developing-applications/building-blocks/pubsub/pubsub-overview/) and FastAPI, `Floki` enables agents to collaborate dynamically in decentralized systems.

## Prerequisites

Before you start, ensure you have the following:

* [Floki environment set up](https://cyb3rward0g.github.io/floki/home/installation/), including Python 3.8 or higher and Dapr CLI.
* Docker installed and running.
* Basic understanding of microservices and event-driven architecture.

## Project Structure

The project is organized into multiple services, each representing an agent or a workflow. Here’s the layout:

```
├── components/              # Dapr configuration files
│   ├── statestore.yaml      # State store configuration
│   ├── pubsub.yaml          # Pub/Sub configuration
├── services/                # Directory for services
│   ├── hobbit/              # Hobbit Agent Service
│   │   └── app.py           # FastAPI app for Hobbit
│   ├── wizard/              # Wizard Agent Service
│   │   └── app.py           # FastAPI app for Wizard
│   ├── elf/                 # Elf Agent Service
│   │   └── app.py           # FastAPI app for Elf
│   ├── workflow-roundrobin/ # Workflow Service
│       └── app.py           # Orchestrator Workflow
├── dapr.yaml                # Multi-App Run Template
```

## Running the Services

1. Multi-App Run: Use the dapr.yaml file to start all services simultaneously:

```bash
dapr run -f .
```

2. Verify console Logs: Each service outputs logs to confirm successful initialization.

![](../../docs/img/workflows_roundrobin_agent_initialization.png)

3. Verify Redis entries: Access the Redis Insight interface at `http://localhost:5540/`

![](../../docs/img/workflows_roundrobin_redis_agents_metadata.png)

4. Verify your agents are healthy: Check the console logs. You should see the following:

![](../../docs/img/workflows_roundrobin_agents_health.png)

## Starting the Workflow

Send an HTTP POST request to the workflow service to start the workflow. Use curl or any API client:

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

The workflow will trigger agents in a round-robin sequence to process the message.

## Monitoring Workflow Execution

1. Check console logs to trace activities in the workflow.

![](../../docs/img/workflows_roundrobin_console_logs_activities.png)

2. Verify Redis entries: Access the Redis Insight interface at `http://localhost:5540/`

![](../../docs/img/workflows_roundrobin_redis_broadcast_channel.png)

3. As mentioned earlier, when we ran dapr init, Dapr initialized, a `Zipkin` container instance, used for observability and tracing. Open `http://localhost:9411/zipkin/` in your browser to view traces > Find a Trace > Run Query.

![](../../docs/img/workflows_roundrobin_zipkin_portal.png)

4. Select the trace entry with multiple spans labeled `<workflow name>: /taskhubsidecarservice/startinstance.`. When you open this entry, you’ll see details about how each task or activity in the workflow was executed. If any task failed, the error will also be visible here.

![](../../docs/img/workflows_roundrobin_zipkin_spans.png)

5. Check console logs to validate if workflow was executed successfuly.

![](../../docs/img/workflows_roundrobin_console_logs_complete.png)

## Customizing the Workflow

The default setup uses the [workflow-roundrobin service](services/workflow-roundrobin/app.py), which processes agent tasks in a `round-robin` order. However, you can easily switch to a different workflow type by updating the `dapr.yaml` file.

### Available Workflow Options

* **RoundRobin**: Cycles through agents in a fixed order, ensuring each agent gets an equal opportunity to process tasks.
* **Random**: Selects an agent randomly for each task.
* **LLM-based**: Uses a large language model (e.g., GPT-4o) to determine the most suitable agent based on the message and context.

### Switching to the LLM-based Workflow

1. Set Up Environment Variables: Create an `.env` file to securely store your API keys and other sensitive information. For example:

```
OPENAI_API_KEY="your-api-key"
OPENAI_BASE_URL="https://api.openai.com/v1"
```

2. Update dapr.yaml: Modify the appDirPath for the workflow service to point to the workflow-llm directory:

```yaml
- appId: WorkflowApp
  appDirPath: ./services/workflow-llm/
  appPort: 8004
  command: ["python3", "app.py"]
  daprGRPCPort: 50004
```

3. Load Environment Variables: Ensure your service script uses Python-dotenv to load these variables automatically:

```python
from dotenv import load_dotenv
load_dotenv()  # Load variables from .env
```

With these updates, the workflow will use the `LLM` to intelligently decide which agent to activate.

### Reset Redis Database

1. Access the Redis Insight interface at `http://localhost:5540/`
2. In the search bar type `*` to select all items in the database.
3. Click on `Bulk Actions` > `Delete` > `Delete`

![](../../docs/img/workflows_roundrobin_redis_reset.png)

You should see an empty database now:

![](../../docs/img/workflows_roundrobin_redis_empty.png)

### Testing the LLM-based Workflow

Restart the services with `dapr run -f` . and send a message to the workflow. Ensure your `.env` file is configured correctly and contains the necessary credentials.
