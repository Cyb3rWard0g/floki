from floki.types import AgentError, AssistantMessage, ChatCompletion, ToolMessage
from floki.agent import AgentBase
from typing import List, Optional, Dict, Any, Union
from pydantic import Field, ConfigDict
import logging

logger = logging.getLogger(__name__)

class ToolCallAgent(AgentBase):
    """
    Agent that manages tool calls and conversations using a language model.
    It integrates tools and processes them based on user inputs and task orchestration.
    """

    tool_history: List[ToolMessage] = Field(default_factory=list, description="Executed tool calls during the conversation.")
    tool_choice: Optional[str] = Field(default=None, description="Strategy for selecting tools ('auto', 'required', 'none'). Defaults to 'auto' if tools are provided.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Initialize the agent's settings, such as tool choice and parent setup.
        Sets the tool choice strategy based on provided tools.
        """
        self.tool_choice = self.tool_choice or ('auto' if self.tools else None)
        
        # Proceed with base model setup
        super().model_post_init(__context)

    def run(self, input_data: Optional[Union[str, Dict[str, Any]]] = None) -> Any:
        """
        Executes the agent's main task using the provided input or memory context.

        Args:
            input_data (Optional[Union[str, Dict[str, Any]]]): User's input, either as a string, a dictionary, or `None` to use memory context.

        Returns:
            Any: The agent's response after processing the input.

        Raises:
            AgentError: If the input data is invalid or if a user message is missing.
        """
        logger.debug(f"Agent run started with input: {input_data if input_data else 'Using memory context'}")

        # Format messages; construct_messages already includes chat history.
        messages = self.construct_messages(input_data or {})

        # Get Last User Message
        user_message = self.get_last_user_message(messages)
        
        if input_data:
            # Add the new user message to memory only if input_data is provided
            if user_message:  # Ensure a user message exists before adding to memory
                self.memory.add_message(user_message)

        # Always print the last user message for context, even if no input_data is provided
        if user_message:
            self.text_formatter.print_message(user_message)

        # Process conversation iterations
        return self.process_iterations(messages)

    def process_response(self, tool_calls: List[dict]) -> None:
        """
        Execute tool calls and log their results in the tool history.

        Args:
            tool_calls (List[dict]): Definitions of tool calls from the response.

        Raises:
            AgentError: If an error occurs during tool execution.
        """
        for tool in tool_calls:
            function_name = tool.function.name
            try:
                logger.info(f"Executing {function_name} with arguments {tool.function.arguments}")
                result = self.tool_executor.execute(function_name, **tool.function.arguments_dict)
                tool_message = ToolMessage(tool_call_id=tool.id, name=function_name, content=str(result))
                
                self.text_formatter.print_message(tool_message)
                self.tool_history.append(tool_message)
            except Exception as e:
                logger.error(f"Error executing tool {function_name}: {e}")
                raise AgentError(f"Error executing tool '{function_name}': {e}") from e
    
    def process_iterations(self, messages: List[Dict[str, Any]]) -> Any:
        """
        Processes conversation iterations, invoking tool calls as needed.

        Args:
            messages (List[Dict[str, Any]]): Initial conversation messages.

        Returns:
            Any: The final response content after processing all iterations.

        Raises:
            AgentError: If an error occurs during chat generation or if maximum iterations are reached.
        """
        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations} started.")
            
            messages += self.tool_history

            try:
                response: ChatCompletion = self.llm.generate(messages=messages, tools=self.tools, tool_choice=self.tool_choice)
                response_message = response.get_message()
                self.text_formatter.print_message(response_message)

                if response.get_reason() == "tool_calls":
                    self.tool_history.append(response_message)
                    self.process_response(response.get_tool_calls())
                else:
                    self.memory.add_message(AssistantMessage(response.get_content()))
                    self.tool_history.clear()
                    return response.get_content()

            except Exception as e:
                logger.error(f"Error during chat generation: {e}")
                raise AgentError(f"Failed during chat generation: {e}") from e

        logger.info("Max iterations reached. Agent has stopped.")