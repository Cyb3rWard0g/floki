# Floki: Agentic Workflows Made Simple

[![pypi](https://img.shields.io/pypi/v/floki-ai.svg)](https://pypi.python.org/pypi/floki-ai)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/floki-ai)](https://pypi.org/project/floki-ai/)
[![GitHub Repo stars](https://img.shields.io/github/stars/Cyb3rWard0g/floki)](https://github.com/Cyb3rWard0g/floki)
[![license](https://img.shields.io/github/license/Cyb3rWard0g/floki.svg)](https://github.com/Cyb3rWard0g/floki/blob/main/LICENSE)

![](logo-workflows.png)

Floki is an open-source framework for researchers and developers to experiment with LLM-based autonomous agents. It provides tools to create, orchestrate, and manage agents while seamlessly connecting to LLM inference APIs. Built on Dapr, Floki leverages a unified programming model that simplifies microservices and supports both deterministic workflows and event-driven interactions. Using Dapr’s Virtual Actor pattern, Floki enables agents to function as independent, self-contained units that process messages sequentially, eliminating concurrency concerns while seamlessly integrating into larger workflows. It also facilitates agent collaboration through Dapr’s Pub/Sub integration, where agents communicate via a shared message bus, simplifying the design of workflows where tasks are distributed efficiently, and agents work together to achieve shared goals. By bringing together these features, Floki provides a powerful way to explore agentic workflows and the components that enable multi-agent systems to collaborate and scale, all powered by Dapr.

## Why Dapr 🎩?

Dapr provides Floki with a unified programming model that simplifies the development of resilient and scalable systems by offering built-in APIs for features such as service invocation, Pub/Sub messaging, workflows, and even state management. These components, essential for defining agentic workflows, allow developers to focus on designing agents and workflows rather than rebuilding foundational features. By leveraging Dapr’s sidecar architecture and portable, event-driven runtime, Floki also enables agents to collaborate effectively, share tasks, and adapt dynamically across cloud and edge environments. This seamless integration brings together deterministic workflows and LLM-based decision-making into a unified system, making it easier to experiment with multi-agent systems and scalable agentic workflows.

### Key Dapr Features in Floki:
* 🎯 **Service-to-Service Invocation**: Facilitates direct communication between agents with built-in service discovery, error handling, and distributed tracing. Agents can leverage this for synchronous messaging in multi-agent workflows.
* ⚡️ **Publish and Subscribe**: Supports loosely coupled collaboration between agents through a shared message bus. This enables real-time, event-driven interactions critical for task distribution and coordination.
* 🔄 **Workflow API**: Defines long-running, persistent workflows that combine deterministic processes with LLM-based decision-making. Floki uses this to orchestrate complex multi-step agentic workflows seamlessly.
* 🧠 **State Management**: Provides a flexible key-value store for agents to retain context across interactions, ensuring continuity and adaptability during workflows.
* 🤖 **Actors**: Implements the Virtual Actor pattern, allowing agents to operate as self-contained, stateful units that handle messages sequentially. This eliminates concurrency concerns and enhances scalability in Floki's agent systems.

## Getting Started

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Set up in 2 minutes__

    ---

    Install [`Floki`](https://github.com/Cyb3rWard0g/floki) with [`pip`](#) and set up your dapr environment in minutes

    [:octicons-arrow-right-24: Installation](home/installation.md)

-   :material-rocket-launch:{ .lg .middle } __Start experimenting__

    ---

    Build your first agent and design a custom workflow to get started with Floki.

    [:octicons-arrow-right-24: Quickstarts](home/quickstarts/index.md)

-   :material-lightbulb-on:{ .lg .middle } __Learn more__

    ---

    Learn more about Floki and its main components!

    [:octicons-arrow-right-24: Concepts](concepts/agents.md)

-   :material-scale-balance:{ .lg .middle } __Open Source, MIT__

    ---

    Floki is licensed under MIT and available on [GitHub]

    [:octicons-arrow-right-24: License](https://github.com/Cyb3rWard0g/floki/blob/main/LICENSE)

</div>
