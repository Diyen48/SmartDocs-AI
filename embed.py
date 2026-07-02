import os
import tempfile
import requests
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
import utils as Utils
import uuid


# ── PDF Loading ───────────────────────────────────────────────────────────────

def load_pdf_from_path(file_path: str) -> list:
    """Load a PDF from a local file path into LangChain Documents."""
    loader = PyPDFLoader(file_path)
    return loader.load()



# ── Splitting ─────────────────────────────────────────────────────────────────

def split_documents(pages: list) -> list:
    """Split pages into smaller overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    return splitter.split_documents(pages)



def _store_chunks(chunks: list, doc_name: str):
    """
    Store all chunks in MongoDB.
    Also save filename and page number.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = Utils.mongo_client
    collection = client[Utils.DATABASE_NAME][Utils.COLLECTION_NAME]

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=Utils.VECTOR_INDEX_NAME,
        text_key="text",
        embedding_key="embedding",
    )

    for chunk in chunks:

        chunk.metadata["filename"] = doc_name

        if "page" not in chunk.metadata:
            chunk.metadata["page"] = 0

    print(f"Embedding {len(chunks)} chunks...")

    vector_store.add_documents(chunks)

    print("Embedding completed.")


# ── Public API ──────────────────────────────────────────────────────────────


def embed_and_store_from_path(file_path: str, doc_name: str,username):

    client = Utils.mongo_client
    docs = client[Utils.DATABASE_NAME][Utils.DOCUMENT_COLLECTION]
    existing = docs.find_one({"filename": doc_name, "username": username})
    if existing:
        return existing["document_id"]
    document_id = str(uuid.uuid4())

    pages = load_pdf_from_path(file_path)

    chunks = split_documents(pages)

    print(f"Document ID : {document_id}")
    print(f"Loaded {len(pages)} pages")
    print(f"Created {len(chunks)} chunks")

    for chunk in chunks:
        chunk.metadata["document_id"] = document_id
        chunk.metadata["filename"] = doc_name
        chunk.metadata["username"] = username

    _store_chunks(chunks, doc_name)

    
    docs.insert_one({
        "document_id": document_id,
        "filename": doc_name,
        "username": username
    })
    return document_id


def get_uploaded_documents(username):

    client = Utils.mongo_client

    collection = client[
        Utils.DATABASE_NAME
    ][Utils.DOCUMENT_COLLECTION]
    docs = {}

    for doc in collection.find({"username":username}, {"_id": 0}):
        docs[doc["filename"]] = doc
    return list(docs.values())

if __name__ == "__main__":
    pass