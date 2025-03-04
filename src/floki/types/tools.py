from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Dict, Literal, Optional, List

class OAIFunctionDefinition(BaseModel):
    """
    Represents a callable function in the OpenAI API format.
    
    This class provides the structure required to define a function that the OpenAI API can call.
    It includes the function's name, a detailed description, and a dictionary specifying its parameters.
    """

    name: str = Field(..., description="The name of the function.")
    description: str = Field(..., description="A detailed description of what the function does.")
    parameters: Dict = Field(..., description="A dictionary describing the parameters that the function accepts.")


class OAIToolDefinition(BaseModel):
    """Represents a tool in the OpenAI API format, such as a function, code interpreter, or file search."""

    type: Literal["function", "code_interpreter", "file_search"] = Field(..., description="Type of the tool.")
    function: Optional[OAIFunctionDefinition] = Field(None, description="Function definition, required if type is 'function'.")

    @field_validator('function')
    def check_function_requirements(cls, v, info: ValidationInfo):
        if info.data.get('type') == 'function' and not v:
            raise ValueError("Function definition must be provided for function type tools.")
        return v

class ClaudeToolDefinition(BaseModel):
    """Represents a callable function in the Anthropic Claude API format."""

    name: str = Field(..., description="Name of the function.")
    description: str = Field(..., description="Description of the function's purpose and usage.")
    input_schema: Dict = Field(..., description="Input schema defining function parameters.")

class GeminiFunctionDefinition(BaseModel):
    """Represents a callable function in the Google Gemini API format."""

    name: str = Field(..., description="Function name (a-z, A-Z, 0-9, underscores, dashes, max length 64).")
    description: str = Field(..., description="Purpose of the function, used by the model to decide invocation.")
    parameters: Dict = Field(..., description="Function parameters in OpenAPI 3.0 JSON Schema format.")

class GeminiToolDefinition(BaseModel):
    """Represents a tool in the Google Gemini API format for function declaration."""

    function_declarations: List[GeminiFunctionDefinition] = Field(..., description="Structured representation of function declarations per OpenAPI 3.0.")