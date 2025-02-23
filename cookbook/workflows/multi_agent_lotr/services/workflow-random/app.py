from floki import RandomOrchestrator
from dotenv import load_dotenv
import asyncio
import logging

async def main():
    try:
        random_workflow_service = RandomOrchestrator(
            name="Orchestrator",
            message_bus_name="messagepubsub",
            agents_registry_store_name="agentstatestore",
            state_store_name="agenticworkflowstate",
            port=8008,
            daprGrpcPort=50008,
            max_iterations=3
        )
        await random_workflow_service.start()
    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(main())