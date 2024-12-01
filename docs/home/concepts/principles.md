# Core Principles

## 1. Powered by Dapr

Floki leverages Dapr’s modular and scalable architecture to provide a foundation that simplifies:

* Service communication: Built-in APIs for service-to-service invocation reduce the complexity of connecting agents and services.
* Cloud integration: Deploy seamlessly across local, cloud, or hybrid environments.
* Event-driven collaboration: Agents communicate effectively through Pub/Sub messaging, enabling dynamic task distribution and real-time updates.

## 2. Modular Component Architecture

Floki inherits Dapr’s pluggable architecture, allowing developers to:

* Swap components like Pub/Sub systems (e.g., RabbitMQ → Kafka) or state stores (e.g., Redis → DynamoDB) without modifying application code.
* Experiment with different setups locally or in the cloud, ensuring flexibility and adaptability.
* Scale effortlessly by relying on modular, production-ready building blocks.

## 3. Virtual Actor Model for Agents

Floki uses Dapr’s Virtual Actor model to enable agents to act as:

* Stateful entities: Agents can retain context across interactions, ensuring continuity in workflows.
* Independent units: Each agent processes tasks sequentially, simplifying concurrency management.
* Scalable services: Agents scale seamlessly across nodes in a cluster, adapting to distributed environments.

## 4. Message-Driven Communication

Floki emphasizes the use of Pub/Sub messaging to empower both event-driven and deterministic workflows. This principle ensures:

* Decoupled Architecture: Agents and services can communicate asynchronously, promoting scalability and flexibility.
* Event-Driven Interactions: Agents respond to real-time events, allowing workflows to adapt dynamically.
* Workflow Enablement: Pub/Sub acts as a backbone for coordinating agents in centralized or decentralized workflows.

## 5. Workflow-Oriented Design

Floki integrates Dapr’s Workflow API to orchestrate:

* Deterministic workflows: Define repeatable processes for agents to follow.
* Dynamic decision-making: Combine rule-based logic with LLM-driven reasoning to handle complex, adaptive tasks.
* Multi-agent collaboration: Coordinate agents across workflows for shared goals and efficient task execution.