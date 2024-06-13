from typing import List
from langchain.document_loaders import UnstructuredURLLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from app.pkg.neo4j.vector import store_data_in_neo4j
from app.pkg.neo4j.credentials import neo4j_credentials
from llama_index.core import SimpleDirectoryReader
from unstructured.partition.auto import partition
from unstructured.cleaners.core import replace_unicode_quotes
from unstructured.chunking.title import chunk_by_title

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_links_data(urls: List[str]):
    """
    Load data from Wikipedia based on the given query.
    """
    # Reading and chunking a Wikipedia article
    # https://neo4j.com/developer-blog/langchain-library-full-support-neo4j-vector-index/
    
    loader = UnstructuredURLLoader(urls=urls)
    raw_documents = loader.load()

    return raw_documents

def process_wikipedia_data(raw_documents):
    """
    Process (chunk and clean) the loaded Wikipedia data.
    """
    # Define chunking strategy
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=20
    )
    logger.info(f"Chunking the documents")
    # Chunk the document
    documents = text_splitter.split_documents(raw_documents)
    # print(f"****\nDocuments:\n{documents}\n****")

    # Remove summary from metadata
    # LangChain’s WikipediaLoader adds a summary to each chunk by default. I thought the added summaries were a bit redundant.
    # For example, if you used a vector similarity search to retrieve the top three results, the summary would be repeated three times.
    # Therefore, I decided to remove it from the dataset.
    # for d in documents:
    #     del d.metadata["summary"]

    return documents

def load_data_from_link_and_store_openai_embeddings_in_neo4j_vector(urls: List[str]):
    try:
        logger.info(
            f"\nLoad data from link and store OpenAI embeddings in a Neo4j Vector\n\tLinks: {urls}\n"
        )

        raw_docs = load_links_data(urls)
        processed_docs = process_wikipedia_data(raw_docs)
        print(f"****\nProcessed docs:\n{processed_docs}\n****")
        store_data_in_neo4j(processed_docs, neo4j_credentials)

    except Exception as e:
        print(f"\n\tAn unexpected error occurred: {e}")
        
def load_data_from_pdf(path: str):
    try:
        logger.info(f"Loading data from PDF: {path}")
        # documents = DirectoryLoader(path).load()
        documents = SimpleDirectoryReader(path).load_data()
        logger.info(f"Raw docs:\n{documents}\n")
        # processed_docs = process_wikipedia_data(documents)
        # print(f"****\nProcessed docs:\n{processed_docs}\n****")
        # store_data_in_neo4j(processed_docs, neo4j_credentials)
        return documents
    except Exception as e:
        logger.error(f"Error loading data from PDF file: {e}")
        
def load_data_from_url(urls: List[str]):
    try:
        logger.info(f"Loading data from URL's: {urls}")
        raw_docs = load_links_data(urls)
        # documents = SimpleDirectoryReader(path).load_data()
        logger.info(f"Raw docs:\n{raw_docs}\n")
        processed_docs = process_wikipedia_data(raw_docs)
        print(f"****\nProcessed docs:\n{processed_docs}\n****")
        # store_data_in_neo4j(processed_docs, neo4j_credentials)
        return processed_docs
    except Exception as e:
        logger.error(f"Error loading data from PDF file: {e}")
        

def load_data_using_unstructured(pdf_content):
    elements = partition(filename= pdf_content, chunking_strategy="basic", chunk_size=4000, multipage_sections=True, strategy= "fast")
    # elements = partition(pdf_content, chunking_strategy="basic", chunk_size=4000, multipage_sections=True)
    for el in elements:
        el.apply(replace_unicode_quotes)
    print("-"*80)
    print(elements)
    print("len(elements)", len(elements))
    for el in elements:
        print(el)
        print("-"*40)

    elements = chunk_by_title(elements, multipage_sections=True, combine_text_under_n_chars=1000, max_characters=4000 )
    print("len(elements-chunked)", len(elements))
    for el in elements:
        print(el)
        print("-"*40)
    
    # print("\n\n".join([str(el) for el in elements]))
    # chunks  = chunk_by_title(elements)
    # print(chunks)
    heading = pdf_content.split('\\')[-1].split('.')[0]
    result = [{"heading": heading, "content": "\n\n".join([str(el) for el in elements])}]
    return result