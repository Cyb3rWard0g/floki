from floki.executors import DockerCodeExecutor
from floki import CodeExecutorAgent
from dotenv import load_dotenv
import asyncio
import logging

async def main():
    try:
        docker_code_executor = DockerCodeExecutor(disable_network_access=False)
        docker_code_executor_agent = CodeExecutorAgent(
            name="DockerCodeExecutorAgent",
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

        await docker_code_executor_agent.start()
    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(main())