from floki.executors import DockerCodeExecutor
from floki import CodeExecutorAgent
from dotenv import load_dotenv
import asyncio
import logging

async def main():
    try:
        # Define Agent
        docker_code_executor = DockerCodeExecutor(disable_network_access=False)
        code_executor_service = CodeExecutorAgent(
            name="CodeExecutorAgent",
            role="Code Executor",
            goal="Execute Python and Bash scripts as provided, returning outputs, errors, and execution details without modification or interpretation",
            message_bus_name="messagepubsub",
            state_store_name="agenticworkflowstate",
            state_key="workflow_state",
            agents_registry_store_name="agentsregistrystore",
            agents_registry_key="agents_registry",
            service_port=8002,
            daprGrpcPort=50002,
            max_iterations=25,
            executor=docker_code_executor
        )

        await code_executor_service.start()
    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(main())