from pydantic import BaseModel, Field, field_validator, ValidationError, model_validator, ConfigDict
from typing import List, Optional, Dict, Literal
import json

class BaseMessage(BaseModel):
    """
    Base class for creating and processing message objects. This class provides common attributes 
    that are shared across different types of messages.
    """
    
    content: Optional[str] = Field(None, description="The main text content of the message. If provided, it initializes the message with this content.")
    role: str = Field(..., description="The role associated with the message (e.g., 'user', 'system', 'assistant'). This must be set by derived classes.")
    name: Optional[str] = Field(None, description="An optional name identifier for the message, useful for distinguishing messages in a conversation.")

    def __init__(self, text: Optional[str] = None, **data):
        """
        Initializes a new BaseMessage instance. If 'text' is provided, it initializes the 'content' attribute with this value.

        Args:
            text (Optional[str]): Text content for the 'content' attribute.
            **data: Additional fields that can be set during initialization, passed as keyword arguments.
        """
        super().__init__(content=text, **data) if text is not None else super().__init__(**data)
    
    @model_validator(mode='after')
    def remove_empty_name(self):
        attrList = []
        for attribute in self.__dict__:
            if attribute == "name":
                if self.__dict__[attribute] is None:
                    attrList.append(attribute)

        for item in attrList:
            delattr(self, item)

        return self

class FunctionCall(BaseModel):
    """
    Represents a function call with its name and arguments, which are stored as a JSON string.
    """
    
    name: str = Field(..., description="The name of the function being invoked.")
    arguments: str = Field(..., description="A JSON string containing the arguments for the function call.")

    @field_validator('arguments', mode='before')
    @classmethod
    def validate_json(cls, v):
        """
        Ensures that the arguments are stored as a JSON string. If a dictionary is provided,
        it converts it to a JSON string. If a string is provided, it validates whether it's a proper JSON string.

        Args:
            v (Union[str, dict]): The JSON string or dictionary of arguments to validate and convert.

        Raises:
            ValueError: If the provided string is not valid JSON or if a type other than str or dict is provided.

        Returns:
            str: The JSON string representation of the arguments.
        """
        if isinstance(v, dict):
            try:
                return json.dumps(v)
            except TypeError as e:
                raise ValidationError(f"Invalid data type in dictionary: {e}")
        elif isinstance(v, str):
            try:
                json.loads(v)  # This is to check if it's valid JSON
                return v
            except json.JSONDecodeError as e:
                raise ValidationError(f"Invalid JSON format: {e}")
        else:
            raise TypeError(f"Unsupported type for field: {type(v)}")
    
    @property
    def arguments_dict(self):
        """
        Property to safely return arguments as a dictionary.
        """
        return json.loads(self.arguments) if self.arguments else {}

class ToolCall(BaseModel):
    """Represents a tool call within a message, detailing the tool that should be called."""
    
    id: str = Field(..., description="A unique identifier for the tool call.")
    type: Literal["function"] = Field(..., description="The type of tool being called. Currently, only 'function' is supported.")
    function: FunctionCall = Field(..., description="The function that should be called as part of the tool call.")

class MessageContent(BaseMessage):
    """
    Extends BaseMessage to include dynamic optional fields for tool and function calls.
    Utilizes post-initialization validation to dynamically manage the inclusion of `tool_calls` and `function_call` fields based on their presence in the initialization data. Fields are only retained if they contain data, thus preventing serialization or display of `None` values, which helps maintain clean and concise object representations.
    """
    
    tool_calls: Optional[List[ToolCall]] = Field(default=None, description="A list of tool calls included in the message, if applicable.")
    function_call: Optional[FunctionCall] = Field(default=None, description="A single function call included in the message, if applicable.")

    @model_validator(mode='after')
    def remove_empty_calls(self):
        attrList = []
        for attribute in self.__dict__:
            if attribute in ('tool_calls','function_call'):
                if self.__dict__[attribute] is None:
                    attrList.append(attribute)

        for item in attrList:
            delattr(self, item)

        return self

class Choice(BaseModel):
    """Represents a choice made by the model, detailing the reason for completion, its index, and the message content."""
    
    finish_reason: str = Field(..., description="Reason why the model stopped generating text.")
    index: int = Field(..., description="Index of the choice in a list of potential choices.")
    message: MessageContent = Field(..., description="Content of the message chosen by the model.")
    logprobs: Optional[Dict] = Field(default=None, description="Log probabilities associated with the choice, if available.")

class ChatCompletion(BaseModel):
    """Represents the full response from the chat API, including all choices, metadata, and usage information."""
    
    choices: List[Choice] = Field(..., description="List of choices provided by the model.")
    created: int = Field(..., description="The Unix timestamp (in seconds) of when the chat completion was created.")
    id: Optional[str] = Field(default=None, description="A unique identifier for the chat completion.")
    model: str = Field(..., description="Model used for generating the response.")
    object: Optional[str] = Field(default=None, description="Type of object returned.")
    usage: Dict = Field(..., description="Information about API usage for this request.")
    
    def get_message(self) -> Optional[str]:
        """
        Retrieve the main message content from the first choice.
        """
        return self.choices[0].message.model_dump() if self.choices else None
    
    def get_reason(self) -> Optional[str]:
        """
        Retrieve the reason for completion from the first choice.
        """
        return self.choices[0].finish_reason if self.choices else None

    def get_tool_calls(self) -> Optional[List[ToolCall]]:
        """
        Retrieve tool calls from the first choice, if available.
        """
        return self.choices[0].message.tool_calls if self.choices and self.choices[0].message.tool_calls else None
    
    def get_content(self) -> Optional[str]:
        """
        Retrieve the content from the first choice's message.
        """
        message = self.get_message()
        return message.get("content") if message else None

class SystemMessage(BaseMessage):
    """Represents a system message, automatically assigning the role to 'system'."""
    
    role: Literal['system'] = Field(default='system', description="The role of the message, set to 'system' by default.")

class UserMessage(BaseMessage):
    """Represents a user message, automatically assigning the role to 'user'."""
    
    role: Literal['user'] = Field(default='user', description="The role of the message, set to 'user' by default.")

class AssistantMessage(BaseMessage):
    """
    Represents an assistant message, potentially including tool calls, automatically assigning the role to 'assistant'.
    This message type is commonly used for responses generated by an assistant.
    """
    
    role: Literal['assistant'] = Field(default='assistant', description="The role of the message, set to 'assistant' by default.")
    tool_calls: Optional[List[ToolCall]] = Field(default=None, description="A list of tool calls included in the response, if any.")
    function_call: Optional[FunctionCall] = Field(default=None, description="A function call included in the response, if any.")

    @model_validator(mode='after')
    def remove_empty_calls(self):
        attrList = []
        for attribute in self.__dict__:
            if attribute in ('tool_calls','function_call'):
                if self.__dict__[attribute] is None:
                    attrList.append(attribute)

        for item in attrList:
            delattr(self, item)
        
        return self

class ToolMessage(BaseMessage):
    """
    Represents a message specifically used for carrying tool interaction information, automatically assigning the role to 'tool'.
    """

    role: Literal['tool'] = Field(default='tool', description="The role of the message, set to 'tool' by default.")
    tool_call_id: str = Field(..., description="Identifier for the specific tool call associated with the message.")

class AssistantFinalMessage(BaseModel):
    """
    Represents a custom final message from the assistant, encapsulating a conclusive response to the user.
    """

    prompt: str = Field(..., description="The initial prompt that led to the final answer.")
    final_answer: str = Field(..., description="The definitive answer or conclusion provided by the assistant.")

class MessagePlaceHolder(BaseModel):
    """
    A placeholder for a list of messages in the prompt template.

    This allows dynamic insertion of message lists into the prompt, such as chat history or 
    other sequences of messages.
    """

    variable_name: str = Field(..., description="The name of the variable representing the list of messages.")

    model_config = ConfigDict(frozen=True)
    
    def __repr__(self):
        return f"MessagePlaceHolder(variable_name={self.variable_name})"

class EventMessageMetadata(BaseModel):
    """
    Represents CloudEvent metadata for describing event context and attributes.

    This class encapsulates core attributes as defined by the CloudEvents specification.
    Each field corresponds to a CloudEvent context attribute, providing additional metadata
    about the event.
    """

    id: Optional[str] = Field(None, description="Unique event identifier. Must be non-empty and combined with 'source' to ensure uniqueness.")
    datacontenttype: Optional[str] = Field(None, description="Content type of the event data value, e.g., 'application/json'. Must follow RFC 2046.")
    pubsubname: Optional[str] = Field(None, description="Name of the Pub/Sub system delivering the event. Implementation-specific.")
    source: Optional[str] = Field(None, description="Identifies the context where the event occurred. Must be a non-empty URI-reference.")
    specversion: Optional[str] = Field(None, description="Version of the CloudEvents specification used. Required and must be non-empty.")
    time: Optional[str] = Field(None, description="Timestamp of when the event occurred, formatted in RFC 3339.")
    topic: Optional[str] = Field(None, description="Name of the topic categorizing the event within the Pub/Sub system.")
    traceid: Optional[str] = Field(None, description="Identifier for tracing systems to correlate events.")
    traceparent: Optional[str] = Field(None, description="Parent identifier in the tracing system, adhering to the W3C Trace Context standard.")
    type: Optional[str] = Field(None, description="Describes the type of event. Required and must be a non-empty string.")
    tracestate: Optional[str] = Field(None, description="Vendor-specific tracing information following the W3C Trace Context standard.")
    headers: Optional[Dict[str, str]] = Field(None, description="HTTP headers or transport metadata as key-value pairs.")