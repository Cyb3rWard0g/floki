# Floki Quickstarts

Dive into our Floki quickstarts to explore core features with practical code samples, designed to get you up and running quickly. From setup to hands-on examples, these resources are your first step into the world of Floki.

!!! info
    Not all quickstarts require Docker, but it is recommended to have your [local Dapr environment set up](../installation.md) with Docker for the best development experience and to follow the steps in this guide seamlessly.

## Quickstarts

| Scenario | Description |
| --- | --- |
| [LLM Inference Client](llm.md) | Learn how to set up and use Floki's LLM Inference Client to interact with language models like OpenAI's `gpt-4o`. This quickstart covers initializing the OpenAIChatClient, managing environment variables, and generating structured responses using Pydantic models. |
| [LLM-based AI Agents](agents.md) | Discover how to create LLM-based autonomous agents. This quickstart walks you through defining tools with Pydantic schemas, setting up agents with clear roles and goals, and enabling dynamic task execution using OpenAI's Function Calling. |
| [Dapr & Floki Workflows](dapr_workflows.md) | Explore how Floki builds on Dapr workflows to simplify long-running process management. Learn how to define tasks, integrate tools, and add LLM reasoning to extend workflow capabilities. |
| [LLM-based Task Workflows](llm_workflows.md) | Design structured, step-by-step workflows with LLMs providing reasoning at key stages. This quickstart covers task orchestration with Python functions and integrating LLM Inference APIs. |
| [Event-Driven Agentic Workflows](agentic_workflows.md) | Leverage event-driven systems with pub/sub messaging to enable agents to collaborate dynamically. This quickstart demonstrates setting up workflows for decentralized, real-time agent interaction. |