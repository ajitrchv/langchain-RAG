import os
import sys
from pathlib import Path
from typing import Any, Dict

# Make project root importable (ingestion.py, logger.py) when run from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeSparseVectorStore
from langchain_openai import OpenAIEmbeddings

from ingestion import initPineCone

load_dotenv()

vStore = initPineCone("langchain-doc-index")

model = init_chat_model("gpt-5.2", model_provider="openai")

@tool(response_format="content_and_artifact")
def retrieve_context(query):
    """Retrieve relevant documentation to help answer the queries about langchain"""
    retrieved_docs = vStore.as_retriever().invoke(query, k=4)
    serialized = "\n\n".join(
        (f"source: {doc.metadata.get('source', 'unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def run_llm(query) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentaion.
    Args:
        query: The users question

    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    system_prompt=(
        "You are a helpful AI assistant that answers questions about langchain documentaion"
        "You have access to a tool that retrieves relevant docs"
        "Use the tool to find relevant info before answering questions"
        "Always cite the sources you use in your answers"
        "If you cannot find the user in the retrieved document, say so."
    )
    agent= create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    messages: list[AnyMessage | dict[str, Any]] = [{"role":"user", "content":query}]
    response = agent.invoke({"messages": messages})
    answer = response["messages"][-1].content
    context_docs=[]
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    return{
        "answer": answer,
        "context": context_docs,
    }  


if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)
    




