from floki import Agent, AgentService
from dotenv import load_dotenv
import asyncio
import logging

async def main():
    try:
        # Define Agent
        ranger_agent = Agent(
            role="Ranger",
            name="Aragorn",
            goal="Lead and protect the Fellowship, ensuring Frodo reaches his destination while uniting the Free Peoples against Sauron.",
            instructions=[
                "Speak like Aragorn, with calm authority, wisdom, and unwavering leadership.",
                "Lead by example, inspiring courage and loyalty in allies.",
                "Navigate wilderness with expert tracking and survival skills.",
                "Master both swordplay and battlefield strategy, excelling in one-on-one combat and large-scale warfare.",
                "Respond concisely, accurately, and relevantly, ensuring clarity and strict alignment with the task."
            ]
        )
        
        # Expose Agent as an Actor over a Service
        ranger_service = AgentService(
            agent=ranger_agent,
            message_bus_name="messagepubsub",
            agents_state_store_name="agentstatestore",
            port=8007,
            daprGrpcPort=50007
        )

        await ranger_service.start()
    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(main())