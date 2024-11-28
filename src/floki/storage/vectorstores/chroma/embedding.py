import os
from typing import Optional
from chromadb.utils import embedding_functions

class ChromaEmbeddingManager:
    def __init__(self, service: str, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the embedding manager with the specified service.

        Args:
            service (str): The embedding service to use ('openai', 'huggingface', or 'sentence-transformers').
            api_key (Optional[str]): The API key for the embedding service. If None, will use environment variable.
            model (Optional[str]): The model name for generating embeddings.
        """
        self.service = service.lower()
        self.api_key = api_key or self._get_api_key()
        self.model = model or self._get_default_model_name()
        self.embedding_function = self._get_embedding_function()

    def _get_api_key(self):
        """
        Get the API key from environment variables based on the service.
        """
        api_key_env = {"openai": "OPENAI_API_KEY", "huggingface": "HF_TOKEN"}
        if self.service in api_key_env:
            api_key = os.getenv(api_key_env[self.service])
            if not api_key:
                raise ValueError(f"{self.service.capitalize()} API key must be provided either as an argument or as an environment variable '{api_key_env[self.service]}'.")
            return api_key
        return None

    def _get_default_model_name(self):
        """
        Get the default model name based on the service.
        """
        default_model_name = {
            "openai": "text-embedding-ada-002",
            "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers": "all-MiniLM-L6-v2"
        }
        if self.service in default_model_name:
            return default_model_name[self.service]
        raise ValueError("Unsupported embedding service. Choose either 'openai', 'huggingface', or 'sentence-transformers'.")

    def _get_embedding_function(self):
        """
        Return the embedding function based on the specified service.
        """
        if self.service == "openai":
            return embedding_functions.OpenAIEmbeddingFunction(api_key=self.api_key, model_name=self.model)
        elif self.service == "huggingface":
            return embedding_functions.HuggingFaceEmbeddingFunction(api_key=self.api_key, model_name=self.model)
        elif self.service == "sentence-transformers":
            return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.model)
        else:
            raise ValueError("Unsupported embedding service. Choose either 'openai', 'huggingface', or 'sentence-transformers'.")