from floki import Agent, AgentService
from dotenv import load_dotenv
import asyncio
import logging

async def main():
    try:
        # Define Agent
        elf_agent = Agent(
            role="Elf",
            name="Legolas",
            goal="Act as a scout, marksman, and protector, using keen senses and deadly accuracy to ensure the success of the journey.",
            instructions=[
                "Speak like Legolas, with grace, wisdom, and keen observation.",
                "Be swift, silent, and precise, moving effortlessly across any terrain.",
                "Use superior vision and heightened senses to scout ahead and detect threats.",
                "Excel in ranged combat, delivering pinpoint arrow strikes from great distances.",
                "Respond concisely, accurately, and relevantly, ensuring clarity and strict alignment with the task."
            ]
        )

        # Expose Agent as an Actor over a Service
        elf_service = AgentService(
            agent=elf_agent,
            message_bus_name="messagepubsub",
            agents_state_store_name="agentstatestore",
            port=8003,
            daprGrpcPort=50003
        )

        await elf_service.start()
    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())