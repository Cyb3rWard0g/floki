from floki.executors import AzContainerAppsCodeExecutor, AzContainerAppsSessionPools
from floki import CodeExecutorAgent
from dotenv import load_dotenv
import asyncio
import logging

async def main():
    try:
        # Get Environment Variables
        import os
        AZURE_SUBSCRIPTION_ID =os.getenv("AZURE_SUBSCRIPTION_ID")
        
        # Initialize session pool manager
        session_manager = AzContainerAppsSessionPools(
            subscription_id=AZURE_SUBSCRIPTION_ID,
            resource_group="floki-sessionpool",
            session_pool_name="floki-session-pool",
            location="westus2"
        )
        
        # Retrieve or create the session pool
        session_pool = session_manager.get_or_create_session_pool()

        # Retrieve the session pool management endpoint
        pool_management_endpoint = session_pool.pool_management_endpoint
        
        # Initialize the Azure Container Code Executor
        aca_code_executor = AzContainerAppsCodeExecutor(
            pool_management_endpoint=pool_management_endpoint,
            api_version="2024-10-02-preview"
        )

        aca_code_executor_agent = CodeExecutorAgent(
            name="ACACodeExecutorAgent",
            message_bus_name="messagepubsub",
            state_store_name="agenticworkflowstate",
            state_key="workflow_state",
            agents_registry_store_name="agentsregistrystore",
            agents_registry_key="agents_registry",
            service_port=8003,
            daprGrpcPort=50003,
            executor=aca_code_executor
        )

        await aca_code_executor_agent.start()
    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(main())