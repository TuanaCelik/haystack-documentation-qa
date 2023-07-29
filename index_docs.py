import os
from dotenv import load_dotenv
from haystack import Pipeline
from haystack.document_stores import OpenSearchDocumentStore
from haystack.nodes import EmbeddingRetriever, PreProcessor, MarkdownConverter
from readmedocs_fetcher_haystack import ReadmeDocsFetcher
load_dotenv()
README_API_KEY = os.getenv('README_API_KEY')

document_store = OpenSearchDocumentStore()
document_store.delete_documents()

markdown_converter = MarkdownConverter(remove_code_snippets=False)
docs_fetcher = ReadmeDocsFetcher(api_key=README_API_KEY, markdown_converter=markdown_converter, base_url="https://docs.haystack.deepset.ai")
preprocessor = PreProcessor()
retriever = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L12-v2", devices=["mps"], top_k=5)

indexing_pipeline = Pipeline()

indexing_pipeline.add_node(component=docs_fetcher, name="DocsFetcher", inputs=["File"])
indexing_pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["DocsFetcher"])
indexing_pipeline.add_node(component=retriever, name="Retriever", inputs=["PreProcessor"])
indexing_pipeline.add_node(component=document_store, name="DocStore", inputs=["Retriever"])
indexing_pipeline.run()
