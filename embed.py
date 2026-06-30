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


def load_pdf_from_url(url: str) -> list:
    """
    Download a PDF from a URL and load it as LangChain Documents.
    Raises a clear error if the download fails or returns a non-PDF.
    """
    print(f"Downloading PDF from: {url}")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()

    # Validate we actually got a PDF (not a 403 HTML/text response)
    if not response.content or not response.content.startswith(b"%PDF"):
        raise ValueError(
            f"\n\nPDF download failed — server returned {len(response.content)} bytes "
            f"instead of a PDF.\n"
            f"This usually means the site is blocking automated downloads.\n\n"
            f"Fix: Download the PDF manually from:\n  {url}\n"
            f"Then place it in your project folder and call:\n"
            f"  embed.embed_and_store_from_path('eu_ai_act.pdf', 'EU AI Act')\n"
        )

    # PyPDFLoader needs a file path, so save to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    try:
        pages = load_pdf_from_path(tmp_path)
    finally:
        os.remove(tmp_path)

    return pages


# ── Splitting ─────────────────────────────────────────────────────────────────

def split_documents(pages: list) -> list:
    """Split pages into smaller overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    return splitter.split_documents(pages)


# ── MongoDB helpers ───────────────────────────────────────────────────────────

def is_already_embedded() -> bool:
    """Return True if documents already exist in MongoDB (skip re-embedding)."""
    try:
        client = Utils.mongo_client
        col = client[Utils.DATABASE_NAME][Utils.COLLECTION_NAME]
        return col.count_documents({}) > 0
    except Exception as e:
        print(f"MongoDB check failed: {e}")
        return False


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


# ── Public API ────────────────────────────────────────────────────────────────

def embed_and_store(url: str, doc_name: str):
    """Load PDF from URL, split, and store in MongoDB. Main entry point."""
    pages  = load_pdf_from_url(url)
    chunks = split_documents(pages)
    print(f"Loaded {len(pages)} pages → {len(chunks)} chunks")
    _store_chunks(chunks, doc_name)


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

def embed_uploaded_pdf(file_path: str):
    pages = load_pdf_from_path(file_path)

    chunks = split_documents(pages)

    print(f"Loaded {len(pages)} pages")
    print(f"Created {len(chunks)} chunks")

    _store_chunks(chunks, os.path.basename(file_path))

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
    embed_and_store(Utils.EUROPEAN_ACT_URL, "EU AI Act")