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

from ingestion import INDEX_NAME, initPineCone, pdf_namespace

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

def make_pdf_retriever_tool(session_id: str):
    """Build a retrieval tool restricted to the PDFs uploaded in one chat session."""
    pdf_store = initPineCone(INDEX_NAME, namespace=pdf_namespace(session_id))

    @tool(response_format="content_and_artifact")
    def retrieve_pdf_context(query: str):
        """Retrieve relevant passages from the PDF file(s) the user uploaded in this chat"""
        retrieved_docs = pdf_store.as_retriever(search_kwargs={"k": 4}).invoke(query)
        serialized = "\n\n".join(
            f"source: {doc.metadata.get('source', 'unknown')} (page {doc.metadata.get('page', '?')})"
            f"\n\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    return retrieve_pdf_context


def run_llm(query, session_id: str | None = None, use_pdf: bool = False) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentaion.
    Args:
        query: The users question
        session_id: Chat session id, used to locate uploaded PDFs
        use_pdf: When True, answer only from the PDF(s) uploaded in this session

    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    if use_pdf and session_id:
        system_prompt = (
            "You are a helpful AI assistant that answers questions about a PDF file the user uploaded. "
            "You have access to a tool that retrieves passages from that PDF. "
            "Always call the tool before answering. "
            "Answer ONLY from the retrieved passages and never use outside knowledge. "
            "Cite the source file and page number you used. "
            "If the answer is not in the retrieved passages, say that the uploaded file does not contain it."
        )
        tools = [make_pdf_retriever_tool(session_id)]
    else:
        system_prompt=(
            "You are a helpful AI assistant that answers questions about langchain documentaion"
            "You have access to a tool that retrieves relevant docs"
            "Use the tool to find relevant info before answering questions"
            "Always cite the sources you use in your answers"
            "If you cannot find the user in the retrieved document, say so."
        )
        tools = [retrieve_context]
    agent= create_agent(model, tools=tools, system_prompt=system_prompt)

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
    




