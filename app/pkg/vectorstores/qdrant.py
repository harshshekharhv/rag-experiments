
"""
Qdrant client for accessing Qdrant DB
"""

from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from pydantic import BaseModel
from typing import Optional

class QdrantConfig(BaseModel):
    url: Optional[str]
    host: Optional[str]
    port: Optional[int]


class QdrantVectorStore:
    def __init__(self, url: str, collection_name: str) -> QdrantVectorStore:
        client = qdrant_client.QdrantClient(
            # you can use :memory: mode for fast and light-weight experiments,
            # it does not require to have Qdrant deployed anywhere
            # but requires qdrant-client >= 1.1.1
            # location=":memory:"
            # otherwise set Qdrant instance address with:
            # url="http://<host>:<port>"
            # otherwise set Qdrant instance with host and port:
            # host="localhost",
            # port=6333
            # set API KEY for Qdrant Cloud
            # api_key="<qdrant-api-key>",
            url=url
        )
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        return vector_store