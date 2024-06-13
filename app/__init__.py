import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from typing import List, Literal
from app.pkg.document_loaders import load_data_from_link_and_store_openai_embeddings_in_neo4j_vector, \
load_data_from_pdf, load_data_from_url, load_data_using_unstructured
# from app.pkg.vectorstores.qdrant import QdrantVectorStore
from app.pkg.llms.local_seldon_wrapper import SeldonCore
from llama_index.core import PropertyGraphIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
    SchemaLLMPathExtractor,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
# from llama_index.indices.property_graph import SchemaLLMPathExtractor
# from app.pkg.neo4j.vector import Neo4j
from app.lib.constants import OPEN_AI_SECRET_KEY
from app.lib import constants
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core.node_parser import SentenceSplitter

def run(urls: List[str]):
    """
    run will extract data from urls and embed it into neo4j DB
    """
    logger.info("Running RAG service")
    # load_data_from_link_and_store_openai_embeddings_in_neo4j_vector(urls)
    documents = load_data_from_pdf("./app/data/")
    # documents = load_data_using_unstructured("./app/data/Shreddingservice.pdf")
    # logger.info(f"\n****\nNew docs:\n{documents}\n****")
    # documents = load_data_from_url(urls)
    embed_dim = 1536
    graph_store = Neo4jPGStore(
        username=constants.NEO4J_USERNAME,
        password=constants.NEO4J_PASSWORD,
        url=constants.NEO4J_URI,
    )
    # vec_store = None
    client = qdrant_client.QdrantClient(
        # you can use :memory: mode for fast and light-weight experiments,
        # it does not require to have Qdrant deployed anywhere
        # but requires qdrant-client >= 1.1.1
        # location=":memory:"
        # otherwise set Qdrant instance address with:
        # url="http://<host>:<port>"
        # otherwise set Qdrant instance with host and port:
        host="localhost",
        port=6333
        # set API KEY for Qdrant Cloud
        # api_key="<qdrant-api-key>",
        # url="http://localhost:6333"
    )
    vector_store = QdrantVectorStore(client=client, collection_name="rag-mixtral")
    storage_context = StorageContext.from_defaults(vector_store=vector_store, property_graph_store=graph_store)
    

    # entities = Literal["PERSON", "PLACE", "THING"]
    # relations = Literal["PART_OF", "HAS", "IS_A"]
    # schema = {
    #     "PERSON": ["PART_OF", "HAS", "IS_A"],
    #     "PLACE": ["PART_OF", "HAS"], 
    #     "THING": ["IS_A"],
    # }

    # kg_extractor = SchemaLLMPathExtractor(
    #     llm=OpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key=OPEN_AI_SECRET_KEY), 
    #     possible_entities=entities, 
    #     possible_relations=relations, 
    #     kg_validation_schema=schema,
    #     strict=True,  # if false, allows values outside of spec
    # )
    transformations = [
        SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    ]
    seldon_url = "https://seldon-mesh.genai.sc.eng.hitachivantara.com"
    llm = SeldonCore(
            repo_id="mixtral-8x7b-instruct-v0-1-gptq-4bit-32g",
            endpoint_url=seldon_url,
            task="text-generation",
            model_kwargs={
                "temperature": 0.1,
                "max_new_tokens": 1000,
                "top_p": 0.15,
                "top_k": 0,
                "repetition_penalty": 1.1,
            }, 
        )

    index = PropertyGraphIndex.from_documents(
        documents,
        # property_graph_store=graph_store,
        transformations=transformations,
        storage_context=storage_context,
        vector_store=vector_store,
        embed_model=OpenAIEmbedding(model_name="text-embedding-3-small", api_key=OPEN_AI_SECRET_KEY),
        # embed_model=SeldonCore(
        #     endpoint_url=seldon_url,
        #     repo_id="all-minilm-l6-v2-st",
        # ),
        kg_extractors=[
            ImplicitPathExtractor(),
            SimpleLLMPathExtractor(
                llm=OpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key=OPEN_AI_SECRET_KEY),
                # llm=llm,
                num_workers=4,
                max_paths_per_chunk=10,
            ),
            # kg_extractor,
        ],
        show_progress=True,
    )
    retriever = index.as_retriever(
        include_text=False,  # include source text, default True
    )

    nodes = retriever.retrieve("What is Shredding service?")

    for node in nodes:
        print(node.text)
    
    query_engine = index.as_query_engine(include_text=True)

    response = query_engine.query("What is Shredding service?")

    print(str(response))
    