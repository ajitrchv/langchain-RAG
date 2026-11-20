import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Make project root importable (ingestion.py, logger.py) when run from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, ToolMessage
from langchain.tools import tool

from ingestion import INDEX_NAME, initPineCone, pdf_namespace

load_dotenv()

DEFAULT_LLM_CONFIG: Dict[str, Any] = {"provider": "openai", "model": "gpt-5.2"}
LLM_TIMEOUT_SECONDS = 120


def build_model(llm_config: Optional[Dict[str, Any]] = None):
    """Create the chat model for the selected provider (OpenAI, Anthropic or Ollama)."""
    config = llm_config or DEFAULT_LLM_CONFIG
    provider = config["provider"]
    kwargs: Dict[str, Any] = {}
    if provider == "ollama":
        kwargs["base_url"] = config["base_url"]
        kwargs["client_kwargs"] = {"timeout": LLM_TIMEOUT_SECONDS}
    else:
        kwargs["timeout"] = LLM_TIMEOUT_SECONDS
        if config.get("api_key"):
            kwargs["api_key"] = config["api_key"]
    return init_chat_model(config["model"], model_provider=provider, **kwargs)


def make_docs_retriever_tool(openai_api_key: Optional[str] = None):
    """Build the retrieval tool over the LangChain documentation index."""
    docs_store = initPineCone(INDEX_NAME, openai_api_key=openai_api_key)

    @tool(response_format="content_and_artifact")
    def retrieve_context(query):
        """Retrieve relevant documentation to help answer the queries about langchain"""
        retrieved_docs = docs_store.as_retriever().invoke(query, k=4)
        serialized = "\n\n".join(
            (f"source: {doc.metadata.get('source', 'unknown')}\n\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    return retrieve_context


def make_pdf_retriever_tool(session_id: str, openai_api_key: Optional[str] = None):
    """Build a retrieval tool restricted to the PDFs uploaded in one chat session."""
    pdf_store = initPineCone(
        INDEX_NAME, namespace=pdf_namespace(session_id), openai_api_key=openai_api_key
    )

    @tool(response_format="content_and_artifact")
    def retrieve_pdf_context(query: str):
        """Retrieve relevant passages from the PDF file(s) and web page(s) the user added in this chat"""
        retrieved_docs = pdf_store.as_retriever(search_kwargs={"k": 4}).invoke(query)
        serialized = "\n\n".join(
            f"source: {doc.metadata.get('source', 'unknown')}"
            + (f" (page {doc.metadata['page']})" if doc.metadata.get("page") is not None else "")
            + f"\n\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    return retrieve_pdf_context


def run_llm(
    query,
    session_id: str | None = None,
    use_pdf: bool = False,
    llm_config: Optional[Dict[str, Any]] = None,
    openai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentaion.
    Args:
        query: The users question
        session_id: Chat session id, used to locate uploaded PDFs
        use_pdf: When True, answer only from the PDF(s) uploaded in this session
        llm_config: Chat model settings (provider, model, api_key/base_url); defaults to OpenAI
        openai_api_key: Key used for embeddings when retrieving; defaults to OPENAI_API_KEY

    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    if use_pdf and session_id:
        system_prompt = (
            "You are a helpful AI assistant that answers questions about PDF file(s) and web page(s) the user added. "
            "You have access to a tool that retrieves passages from them. "
            "Always call the tool before answering. "
            "Answer ONLY from the retrieved passages and never use outside knowledge. "
            "Cite the source file or URL (and page number for PDFs) you used. "
            "If the answer is not in the retrieved passages, say that the added material does not contain it."
        )
        tools = [make_pdf_retriever_tool(session_id, openai_api_key)]
    else:
        system_prompt=(
            "You are a helpful AI assistant that answers questions about langchain documentaion"
            "You have access to a tool that retrieves relevant docs"
            "Use the tool to find relevant info before answering questions"
            "Always cite the sources you use in your answers"
            "If you cannot find the user in the retrieved document, say so."
        )
        tools = [make_docs_retriever_tool(openai_api_key)]
    agent = create_agent(build_model(llm_config), tools=tools, system_prompt=system_prompt)

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
