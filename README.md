# RAG Pipeline for documentation search

## To run:
1. Create a `.env` file and add your `README_API_KEY` and `OPENAI_API_KEY`
2. `pip install -r requirements.txt`
3. Index documents into an `OpenSearchDocumentStore` (currently set to the default local setup) by running `python index_docs.py`
4. Query docs by running `python query_pipeline.py`. This starts a CLI that expects a query.