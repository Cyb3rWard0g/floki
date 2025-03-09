from floki import AssistantAgent, tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import asyncio
import logging

async def main():
    try:

        class GetWeatherSchema(BaseModel):
            location: str = Field(description="location to get weather for")
        
        class JumpSchema(BaseModel):
            distance: str = Field(description="Distance for agent to jump")

        @tool(args_model=GetWeatherSchema)
        def get_weather(location: str) -> str:
            """Get weather information for a specific location."""
            import random
            temperature = random.randint(60, 80)
            return f"{location}: {temperature}F."

        @tool(args_model=JumpSchema)
        def jump(distance: str) -> str:
            """Jump a specific distance."""
            return f"I jumped the following distance {distance}"

        # Define Weather Agent
        weather_agent = AssistantAgent(
            name="WeatherAgent",
            role="Weather Forecaster",
            goal="Provide real-time weather updates, forecasts, and climate insights to help users plan accordingly.",
            instructions=[
                "Retrieve and analyze weather data for specific locations upon request.",
                "Provide current weather conditions, hourly and weekly forecasts.",
                "Offer insights on severe weather alerts, temperature trends, and climate conditions.",
                "Ensure responses are accurate, up-to-date, and user-friendly.",
                "Use structured reporting for clarity, including temperature, humidity, wind speed, and precipitation chances."
            ],
            tools=[get_weather,jump],
            message_bus_name="messagepubsub",
            state_store_name="agenticworkflowstate",
            state_key="workflow_state",
            agents_registry_store_name="agentsregistrystore",
            agents_registry_key="agents_registry",
            service_port=8001,
            daprGrpcPort=50001
        )
        
        await weather_agent.start()
    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(main())