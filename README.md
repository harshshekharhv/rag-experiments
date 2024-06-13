# rag-experiments

This is a package that generates property graph for documents using Llamaindex libraries and stores it into Neo4j DB and the vector embeddings into Qdrant DB. This library leverages OpenAI for graph generation.

## Prerequisites
- Poetry

## Running locally

Run the below commands after cloning this repo and navigating inside the directory.
- poetry install --no-root
- poetry run python -m main