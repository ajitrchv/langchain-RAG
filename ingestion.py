import asyncio
import io
from importlib import metadata
import os
import re
import ssl
from typing import Any, Dict, List
from unittest import result

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, tavily_crawl, tavily_extract, TavilyMap, tavily_map
from langfuse.langchain import CallbackHandler
from numpy import where
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
from openai.types import vector_store
from logger import(Colors, log_error, log_header, log_info, log_success, log_warning)


load_dotenv()
# configure SSL
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] =certifi.where()
os.environ["REQUESTS_CA_BUNDLE"]=certifi.where()


def get_embeddings(openai_api_key: str | None = None) -> OpenAIEmbeddings:
    """Embedding model for the index. Uses the given key, else OPENAI_API_KEY from the environment."""
    kwargs: Dict[str, Any] = {"api_key": openai_api_key} if openai_api_key else {}
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        show_progress_bar=True,
        chunk_size=50,
        retry_min_seconds=10,
        timeout=60,
        **kwargs,
    )


#chroma=Chroma
# vStore = PineconeVectorStore(index_name="langchain-doc-index", embedding=embeddings)
tExtract=TavilyExtract()
tavily_map=TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tCrawl=TavilyCrawl()

INDEX_NAME = "langchain-doc-index"
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_URLS_PER_MESSAGE = 3
_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+")


def pdf_namespace(session_id: str) -> str:
    """Pinecone namespace holding the PDFs uploaded in a chat session."""
    return f"pdf-{session_id}"


def initPineCone(vStoreIndexName, namespace=None, openai_api_key=None):
    pc = Pinecone(
        api_key=os.environ["PINECONE_API_KEY"]
        )

    index_name = vStoreIndexName

    if not pc.has_index(index_name):
        print(f"Creating Pinecone index: {index_name}")

        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    else:
        print("Index already exists. Skipping creation.")

    vStore = PineconeVectorStore(
        index_name=index_name,
        embedding=get_embeddings(openai_api_key),
        namespace=namespace,
    )
    return vStore


def ingest_pdf(file_bytes: bytes, filename: str, session_id: str, openai_api_key=None) -> int:
    """Extract, chunk and index a PDF into the session's own Pinecone namespace.

    Returns the number of chunks indexed. Raises ValueError for invalid PDFs.
    """
    if len(file_bytes) > MAX_PDF_BYTES:
        raise ValueError("PDF is too large (max 20 MB).")
    if not file_bytes.startswith(b"%PDF"):
        raise ValueError("File is not a valid PDF.")

    reader = PdfReader(io.BytesIO(file_bytes))
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported.")

    source = os.path.basename(filename)
    pages = [
        Document(page_content=text, metadata={"source": source, "page": number})
        for number, page in enumerate(reader.pages, start=1)
        if (text := (page.extract_text() or "").strip())
    ]
    if not pages:
        raise ValueError("No extractable text found in the PDF.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    initPineCone(
        INDEX_NAME, namespace=pdf_namespace(session_id), openai_api_key=openai_api_key
    ).add_documents(chunks)
    return len(chunks)


def find_urls(text: str) -> List[str]:
    """Return the distinct http(s) URLs in a chat message, in order of appearance."""
    urls = []
    for match in _URL_PATTERN.findall(text or ""):
        url = match.rstrip(".,;:!?")
        # Drop a trailing bracket only when it is unbalanced, so Wiki_(topic) URLs survive.
        while url and url[-1] in ")]}" and url.count(url[-1]) > url.count({")": "(", "]": "[", "}": "{"}[url[-1]]):
            url = url[:-1]
        urls.append(url)
    return list(dict.fromkeys(urls))


def ingest_url(url: str, session_id: str, openai_api_key=None) -> int:
    """Fetch a web page, chunk it and index it into the session's Pinecone namespace.

    Returns the number of chunks indexed. Raises ValueError for unusable URLs/pages.
    """
    if not url.startswith("https://"):
        raise ValueError("Only https:// URLs are supported.")

    try:
        result = tExtract.invoke({"urls": [url]})
    except Exception as e:
        raise ValueError("The page could not be fetched.") from e

    pages = [
        r for r in (result.get("results") or []) if (r.get("raw_content") or "").strip()
    ] if isinstance(result, dict) else []
    if not pages:
        raise ValueError("No readable content was found at that URL.")

    docs = [
        Document(page_content=r["raw_content"], metadata={"source": r.get("url") or url})
        for r in pages
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    initPineCone(
        INDEX_NAME, namespace=pdf_namespace(session_id), openai_api_key=openai_api_key
    ).add_documents(chunks)
    return len(chunks)


def delete_pdfs(session_id: str, openai_api_key=None) -> None:
    """Remove all PDF/web page vectors indexed for a chat session."""
    initPineCone(
        INDEX_NAME, namespace=pdf_namespace(session_id), openai_api_key=openai_api_key
    ).delete(delete_all=True)

async def main():
    """Main async function to orchestrate the entire process"""
    log_header("DOCUMENTAION INGESTION PIPELINE")
    log_info(
       "TavilyCrawl: starting to crawl docs from https://python.langchain.com/" 
    )
    res = tCrawl.invoke({
        "url":"https://python.langchain.com/",
        "max_depth":10,
        "extract_depth":"advanced",
        # "instructions": "content on ai agents"
    })
    alldocs = [
        Document(
            page_content=result.get("raw_content",""),
            metadata={'source':result.get("url", ""),})
            for result in res.get("results", [])
            if result.get("raw_content")
            ]
    log_success(f"TavilyCrawl: Successfully crawled {len(alldocs)} URLs from doc site!")
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(alldocs)

    log_success(f"Created {len(chunks)} chunks from {len(alldocs)} documents")


    vStore = initPineCone("langchain-doc-index")
    vStore.add_documents(chunks)

    log_success(f"Successfully indexed {len(chunks)} chunks into Pinecone!")
if __name__ == "__main__":
    asyncio.run(main())
