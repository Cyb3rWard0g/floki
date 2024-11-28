from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Iterable, Any, Union
from pydantic import BaseModel, Field, ConfigDict
import logging

logger = logging.getLogger(__name__)

class VectorStoreBase(BaseModel, ABC):
    """Base interface for a vector store."""

    client: Any = Field(default=None, init=False, description="The client to interact with the vector store.")
    embedding_function: Any = Field(default=None, init=False, description="Embedding function to use to embed documents.")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @abstractmethod
    def add(self, documents: Iterable[str], embeddings: Optional[List[List[float]]] = None, metadatas: Optional[List[dict]] = None, **kwargs: Any) -> List[int]:
        """Add documents to the vector store.

        Args:
            documents (Iterable[str]): Strings to add to the vector store.
            embeddings (Optional[List[List[float]]]): The embeddings of the documents to add to vector store.
            metadatas Optional[List[dict]]: List of metadatas associated with the texts.
            kwargs: vector store specific parameters

        Returns:
            List of ids from adding the texts into the vector store.
        """
        pass

    @abstractmethod
    def delete(self, ids: List[int]) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        pass

    @abstractmethod
    def get(self, ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Retrieves items from vector store by IDs. If no IDs are provided, retrieves all items.

        Args:
            ids (Optional[List[str]]): The IDs of the items to retrieve. If None, retrieves all items.

        Returns:
            List[Dict]: A list of dictionaries containing the metadata and documents of the retrieved items.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the vector store.
        """
        pass

    @abstractmethod
    def search_similar(self, query_texts: Optional[Union[List[str], str]] = None, k: int = 4, **kwargs: Any) -> List[Dict]:
        """Search for similar documents and Return metadata of documents most similar to query.

        Args:
            query_texts: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of metadata of Documents most similar to the query.
        """
        pass

    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Embed documents
        """
        pass