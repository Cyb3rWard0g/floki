from floki.types import AgentError, AssistantMessage, ChatCompletion, FunctionCall
from floki.agent import AgentBase
from floki.tool import AgentTool
from typing import List, Dict, Any, Union, Callable, Literal, Optional, Tuple
from datetime import datetime
from pydantic import Field, ConfigDict
import regex, json, textwrap, logging

logger = logging.getLogger(__name__)

class ReActAgent(AgentBase):
    """
    Agent implementing the ReAct (Reasoning-Action) framework for dynamic, few-shot problem-solving by leveraging
    contextual reasoning, actions, and observations in a conversation flow.
    """

    stop_at_token: List[str] = Field(default=["\nObservation:"], description="Token(s) signaling the LLM to stop generation.")
    tools: List[Union[AgentTool, Callable]] = Field(default_factory=list, description="Tools available for the agent, including final_answer.")
    template_format: Literal["f-string", "jinja2"] = Field(default="jinja2", description="The format used for rendering the prompt template.")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def construct_system_prompt(self) -> str:
        """
        Constructs a system prompt in the ReAct reasoning-action format based on the agent's attributes and tools.
        
        Returns:
            str: The structured system message content.
        """
        # Initialize prompt parts with the current date as the first entry
        prompt_parts = [f"# Today's date is: {datetime.now().strftime('%B %d, %Y')}"]

        # Append name if provided
        if self.name:
            prompt_parts.append("## Name\nYour name is {{name}}.")

        # Append role and goal with default values if not set
        prompt_parts.append("## Role\nYour role is {{role}}.")
        prompt_parts.append("## Goal\n{{goal}}.")

        # Append instructions if provided
        if self.instructions:
            prompt_parts.append("## Instructions\n{{instructions}}")

        # Tools section with schema details
        tools_section = "## Tools\nYou have access ONLY to the following tools:\n"
        for tool in self.tools:
            tools_section += f"{tool.name}: {tool.description}. Args schema: {tool.args_schema}\n"
        prompt_parts.append(tools_section.rstrip())  # Trim any trailing newlines from tools_section

        # Additional Guidelines
        additional_guidelines = textwrap.dedent("""
        If you think about using tool, it must use the correct tool JSON blob format as shown below:
        ```
        {
            "name": $TOOL_NAME,
            "arguments": $INPUT
        }
        ```
        """).strip()
        prompt_parts.append(additional_guidelines)

        # ReAct specific guidelines
        react_guidelines = textwrap.dedent("""
        ## ReAct Format
        Thought: Reflect on the current state of the conversation or task. If additional information is needed, determine if using a tool is necessary. When a tool is required, briefly explain why it is needed for the specific step at hand, and immediately follow this with an `Action:` statement to address that specific requirement. Avoid combining multiple tool requests in a single `Thought`. If no tools are needed, proceed directly to an `Answer:` statement.
        Action:
        ```
        {
            "name": $TOOL_NAME,
            "arguments": $INPUT
        }
        ```
        Observation: Describe the result of the action taken.
        ... (repeat Thought/Action/Observation as needed, but **ALWAYS proceed to a final `Answer:` statement when you have enough information**)
        Thought: I now have sufficient information to answer the initial question.
        Answer: ALWAYS proceed to a final `Answer:` statement once enough information is gathered or if the tools do not provide the necessary data.
        
        ### Providing a Final Answer
        Once you have enough information to answer the question OR if tools cannot provide the necessary data, respond using one of the following formats:
        
        1. **Direct Answer without Tools**:
        Thought: I can answer directly without using any tools. Answer: Direct answer based on previous interactions or current knowledge.
        
        2. **When All Needed Information is Gathered**:
        Thought: I now have sufficient information to answer the question. Answer: Complete final answer here.
        
        3. **If Tools Cannot Provide the Needed Information**:
        Thought: The available tools do not provide the necessary information. Answer: Explanation of limitation and relevant information if possible.
                                
        ### Key Guidelines
        - Always Conclude with an `Answer:` statement.
        - Ensure every response ends with an `Answer:` statement that summarizes the most recent findings or relevant information, avoiding incomplete thoughts.
        - Direct Final Answer for Past or Known Information: If the user inquires about past interactions, respond directly with an Answer: based on the information in chat history.
        - Avoid Repetitive Thought Statements: If the answer is ready, skip repetitive Thought steps and proceed directly to Answer.
        - Minimize Redundant Steps: Use minimal Thought/Action/Observation cycles to arrive at a final Answer efficiently.
        - Reference Past Information When Relevant: Use chat history accurately when answering questions about previous responses to avoid redundancy.
        - Progressively Move Towards Finality: Reflect on the current step and avoid re-evaluating the entire user request each time. Aim to advance towards the final Answer in each cycle.

        ## Chat History
        The chat history is provided to avoid repeating information and to ensure accurate references when summarizing past interactions.                               
        """).strip()
        prompt_parts.append(react_guidelines)

        return "\n\n".join(prompt_parts)
    
    def run(self, input_data: Optional[Union[str, Dict[str, Any]]] = None) -> Any:
        """
        Runs the main logic loop for processing the task and executing actions until a result is reached.

        Args:
            input_data (Optional[Union[str, Dict[str, Any]]]): The task or data for the agent to process. If None, relies on memory.

        Returns:
            Any: Final response after processing the task or reaching a final answer.

        Raises:
            AgentError: On errors during chat message processing or action execution.
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

        # Get Tool Names to validate tool selection
        available_tools = self.tool_executor.get_tool_names()

        # Initialize react_loop for iterative reasoning
        react_loop = ""
        
        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations} started.")

            # Check if "react_loop" is already a variable in the template
            if "react_loop" in self.prompt_template.input_variables:
                # If "react_loop" exists as a variable, construct messages dynamically
                iteration_messages = self.construct_messages({"react_loop": react_loop})
            else:
                # Create a fresh copy of original_messages for this iteration
                iteration_messages = [msg.copy() for msg in messages]

                # Append react_loop to the last message (user or otherwise)
                for msg in reversed(iteration_messages):
                    if msg["role"] == "user":
                        msg["content"] += f"\n{react_loop}"
                        break
                else:
                    # Append react_loop to the last message if no user message is found
                    logger.warning("No user message found in the current messages; appending react_loop to the last message.")
                    iteration_messages[-1]["content"] += f"\n{react_loop}"  # Append react_loop to the last message

            try:
                response: ChatCompletion = self.llm.generate(messages=iteration_messages, stop=self.stop_at_token)

                # Parse response into thought, action, and potential final answer
                thought_action, action, final_answer = self.parse_response(response)

                # Print Thought immediately
                self.text_formatter.print_react_part("Thought", thought_action)
                
                if final_answer:  # Direct final answer provided
                    assistant_final_message = AssistantMessage(final_answer)
                    self.memory.add_message(assistant_final_message)
                    self.text_formatter.print_separator()
                    self.text_formatter.print_message(assistant_final_message, include_separator=False)
                    logger.info("Agent provided a direct final answer.")
                    return final_answer
                
                # If there's no action, update the loop and continue reasoning
                if action is None:
                    logger.info("No action specified; continuing with further reasoning.")
                    react_loop += f"Thought:{thought_action}\n"
                    continue  # Proceed to the next iteration
        
                action_name = action["name"]
                action_args = action["arguments"]

                # Print Action
                self.text_formatter.print_react_part("Action", json.dumps(action))

                if action_name in available_tools:
                    logger.info(f"Executing {action_name} with arguments {action_args}")
                    function_call = FunctionCall(**action)
                    execution_results = self.tool_executor.execute(action_name, **function_call.arguments_dict)
                    
                    # Print Observation
                    self.text_formatter.print_react_part("Observation", execution_results)

                    # Update react_loop with the current execution
                    new_content = f"Thought:{thought_action}\nAction:{action}\nObservation:{execution_results}"
                    react_loop += new_content
                    logger.info(new_content)
                else:
                    raise AgentError(f"Unknown tool specified: {action_name}")
            
            except Exception as e:
                logger.error(f"Failed during chat generation: {e}")
                raise AgentError(f"Failed during chat generation: {e}") from e
        
        logger.info("Max iterations completed. Agent has stopped.")
    
    def parse_response(self, response: ChatCompletion) -> Tuple[str, Optional[dict], Optional[str]]:
        """
        Extracts the thought, action, and final answer (if present) from the language model response.

        Args:
            response (ChatCompletion): The language model's response message.

        Returns:
            tuple: (thought content, action dictionary if present, final answer if present)

        Raises:
            ValueError: If the action details cannot be decoded from the response.
        """
        pattern = r'\{(?:[^{}]|(?R))*\}'  # Pattern to match JSON blobs
        message_content = response.get_content()

        # Use regex to find the start of "Action" or "Final Answer" (case insensitive)
        action_split_regex = regex.compile(r'(?i)action:\s*', regex.IGNORECASE)
        final_answer_regex = regex.compile(r'(?i)answer:\s*(.*)', regex.IGNORECASE | regex.DOTALL)
        thought_label_regex = regex.compile(r'(?i)thought:\s*', regex.IGNORECASE)

        # Clean up any repeated or prefixed "Thought:" labels
        message_content = thought_label_regex.sub('', message_content).strip()

        # Check for "Final Answer" directly in the thought
        final_answer_match = final_answer_regex.search(message_content)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip() if final_answer_match.group(1) else None
            return message_content, None, final_answer

        # Split the content into "thought" and "action" parts
        if action_split_regex.search(message_content):
            parts = action_split_regex.split(message_content, 1)
            thought_part = parts[0].strip()  # Everything before "Action" is the thought part
            action_part = parts[1] if len(parts) > 1 else None  # Everything after "Action" is the action part
        else:
            thought_part = message_content
            action_part = None

        # If there's an action part, attempt to extract the JSON blob
        if action_part:
            matches = regex.finditer(pattern, action_part, regex.DOTALL)
            for match in matches:
                try:
                    action_dict = json.loads(match.group())
                    return thought_part, action_dict, None  # Return thought and action directly
                except json.JSONDecodeError:
                    continue

        # If no action is found, just return the thought part with None for action and final answer
        return thought_part, None, None