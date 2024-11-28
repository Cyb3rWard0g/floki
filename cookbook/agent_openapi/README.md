# OpenAPI Agent

The `OpenAPI Agent` represents a specialized agent designed to interact with the external world by transforming OpenAPI specifications into tools. This agent is crucial for scenarios where precise and authenticated API interactions are necessary, allowing the agent to understand and utilize API endpoints dynamically. By leveraging OpenAPI specifications, the agent can adapt to a wide range of APIs, converting each specification into tools that it can use autonomously.

## Agents

| Pattern | Overview |
| --- | --- |
| [ReAct (Reason + Act) MS Graph](react_agent_openapi_msgraph.ipynb) | An OpenAPI agent that applies the `ReAct` prompting technique, following a chain-of-thought reasoning (Thought, Action, Observation) loop. This agent autonomously selects the appropriate MS Graph API endpoint, performs the call, and integrates the response back into its reasoning cycle. |

## Tools
The `OpenAPI Agent` has two main tools created from OpenAPI specifications to facilitate dynamic API interaction. These tools allow the agent to identify relevant API endpoints and execute API calls effectively. Below is a breakdown of each tool's purpose, inputs, and how it operates within the agent's workflow.

### get_openapi_definition

* **Goal**: This tool retrieves a list of relevant API endpoints from OpenAPI specifications that the agent could use to fulfill the user’s query. The tool leverages a vector store to store and search through API definitions, helping the agent narrow down potential APIs based on the task at hand.
* **Functionality**:
    * Similarity Search: Takes the user’s input and queries the `VectorToolStore` to find similar API tools. It ranks potential API endpoints based on similarity to the user’s task and returns the top matches.
    * Tool Usage: This tool is always called before any API call execution to ensure the agent understands which endpoint to use.

### open_api_call_executor
* **Goal**: This tool is responsible for executing API calls using the specific parameters and configuration associated with the selected OpenAPI endpoint. It provides flexibility to adjust API paths, methods, headers, and query parameters, making it versatile for interacting with any OpenAPI-defined API.
* **Functionality**:
    * API Call Execution: Takes in a structured input of HTTP method, path parameters, headers, and other data required to make the API request.
    * Endpoint Selection: After get_openapi_definition suggests possible endpoints, this tool is used to execute the chosen endpoint with specific parameters.
    * Version Management: Ensures the correct API version is used, preventing duplication or misalignment of API path versions.

## How the Tools Work Together?
* Identify Relevant Endpoint: The agent first uses get_openapi_definition to identify a relevant API endpoint based on the user’s query.
* Execute API Call: With the selected endpoint, open_api_call_executor is called to make the actual API request, providing the necessary method, parameters, headers, and data.

This design allows the `OpenAPI Agent` to dynamically interpret and call any API defined within an OpenAPI specification, adapting flexibly to various tasks and user requests.