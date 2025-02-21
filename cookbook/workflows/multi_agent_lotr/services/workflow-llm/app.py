from floki import LLMOrchestrator
from dotenv import load_dotenv
import asyncio
import logging

async def main():
    try:
        agentic_orchestrator = LLMOrchestrator(
            name="Orchestrator",
            message_bus_name="messagepubsub",
            agents_registry_store_name="agentstatestore",
            state_store_name="agenticworkflowstate",
            port=8008,
            daprGrpcPort=50008,
            max_iterations=20
        )

        await agentic_orchestrator.start()
    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(main())