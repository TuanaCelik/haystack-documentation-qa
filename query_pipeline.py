import os
from dotenv import load_dotenv
from haystack import Pipeline
from haystack.nodes import AnswerParser, EmbeddingRetriever, PromptNode, PromptTemplate
from haystack.document_stores import OpenSearchDocumentStore
from haystack.utils import print_answers

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

document_store = OpenSearchDocumentStore()

retriever = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L12-v2", devices=["mps"], top_k=5)

prompt_template = PromptTemplate(prompt="You will be given some technical documentation about Haystack. Answer the query based on the contents of the documentation. Documentation: {join(documents)}\n Query: {query}\n Answer:",
                                 output_parser=AnswerParser())

prompt_node = PromptNode(default_prompt_template=prompt_template, model_name_or_path="gpt-4", api_key=OPENAI_API_KEY)

query_pipeline = Pipeline()
query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])

while True:
    query = input("Ask a question: ")
    result = query_pipeline.run(query=query)
    print_answers(result, details="minimum")