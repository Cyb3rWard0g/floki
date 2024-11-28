# The Weather Agent

The Weather Agent represents a basic example of an agent that interacts with the external world through tools, such as APIs. This agent demonstrates how a language model (LLM) can suggest which tool to use and provide the necessary inputs for tool execution. However, it is the agent—not the language model—that executes the tool and processes the results. Once the tool has been executed, the results are passed back to the language model for further suggestions, summaries, or next actions. This agent showcases the foundational concept of integrating language models with external tools to retrieve real-world data, such as weather information.

## Agents

| Pattern | Overview |
| --- | --- |
| [ToolCall (Function Calling)](toolcall_agent.ipynb) | A weather agent that uses OpenAI’s tool calling (Function Calling) to pass tools in JSON schema format. The language model suggests the tool to be used based on the task, but the agent executes the tool and processes the results. |
| [ReAct (Reason + Act)](react_agent.ipynb) | A weather agent following the ReAct prompting technique. The language model uses a chain-of-thought reasoning process (Thought, Action, Observation) to suggest the next tool to use. The agent then executes the tool, and the results are fed back into the reasoning loop. |

## Tools

* **WeatherTool**: A tool that allows the agent to retrieve weather data by first obtaining geographical coordinates (latitude and longitude) using the Nominatim API. For weather data, the agent either calls the National Weather Service (NWS) API (for locations in the USA) or the Met.no API (for locations outside the USA). This tool is executed by the agent based on the suggestions provided by the language model.
* **HistoricalWeather**: A tool that retrieves historical weather data for a specified location and date range. The agent uses the Nominatim API to get the coordinates for the specified location and calls the Open-Meteo Historical Weather API to retrieve temperature data for past dates. This tool allows the agent to compare past weather conditions with current forecasts, providing richer insights.

### APIs Used

* Nominatim API: Provides geocoding services to convert city, state, and country into geographical coordinates (latitude and longitude).
    * Endpoint: https://nominatim.openstreetmap.org/search.php
    * Purpose: Used to fetch coordinates for a given location, which is then passed to weather APIs.
* National Weather Service (NWS) API: Provides weather data for locations within the United States.
    * Endpoint: https://api.weather.gov
    * Purpose: Used to retrieve detailed weather forecasts and temperature data for locations in the USA.
* Met.no API: Provides weather data for locations outside the United States.
    * Endpoint: https://api.met.no/weatherapi
    * Purpose: Used to retrieve weather forecasts and temperature data for locations outside the USA, offering international coverage.
* Open-Meteo Historical Weather API: Provides historical weather data for any location worldwide.
    * Endpoint: https://archive-api.open-meteo.com/v1/archive
    * Purpose: Used to retrieve historical weather data, including temperature readings for past dates, allowing the agent to analyze past weather conditions and trends.