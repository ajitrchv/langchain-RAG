import asyncio
from importlib import metadata
import os
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
from openai.types import vector_store
from logger import(Colors, log_error, log_header, log_info, log_success, log_warning)


load_dotenv()
# configure SSL
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] =certifi.where()
os.environ["REQUESTS_CA_BUNDLE"]=certifi.where()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=True, chunk_size=50, retry_min_seconds=10
)


#chroma=Chroma
# vStore = PineconeVectorStore(index_name="langchain-doc-index", embedding=embeddings)
tavily_extract=TavilyExtract()
tavily_map=TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tCrawl=TavilyCrawl()

def initPineCone(vStoreIndexName):
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
        embedding=embeddings
    )
    return vStore

async def main():
    """Main async function to orchestrate the entire process"""
    log_header("DOCUMENTAION INGESTION PIPELINE")
    log_info(
       "TavilyCrawl: starting to crawl docs from https://python.langchain.com/" 
    )
    res = tCrawl.invoke({
        "url":"https://python.langchain.com/",
        "max_depth":1,
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
    chunk_size=1000,
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
