from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
import utils as Utils
from agent.memory import ConversationMemory
from langchain_core.documents import Document

class SmartDocsContext:

    def __init__(self,session_id: str,username: str,document_id: str):

        self.session_id = session_id
        self.username = username
        self.document_id = document_id

        self.llm = self._create_llm()
        self.embeddings = self._create_embeddings()
        self.vector_store = self._create_vector_store()
        self.memory = ConversationMemory(session_id)

    def _create_llm(self):
        return ChatGroq(
            groq_api_key=Utils.GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0,
        )

    def _create_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def _create_vector_store(self):
        collection = Utils.mongo_client[
            Utils.DATABASE_NAME
        ][Utils.COLLECTION_NAME]

        return MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=self.embeddings,
            index_name=Utils.VECTOR_INDEX_NAME,
            text_key="text",
            embedding_key="embedding",
        )


    def get_document(self, filename=None):

        collection = Utils.mongo_client[
            Utils.DATABASE_NAME
        ][Utils.COLLECTION_NAME]

        query = {
            "username": self.username,
        }

        if filename:
            query["filename"] = filename
        else:
            query["document_id"] = self.document_id

        cursor = collection.find(query)

        docs = []

        for chunk in cursor:

            docs.append(
                Document(
                    page_content=chunk["text"],
                    metadata={
                        "page": chunk.get("page", 0),
                        "filename": chunk.get("filename", ""),
                        "document_id": chunk.get("document_id"),
                        "username": chunk.get("username"),
                    },
                )
            )

        docs.sort(
            key=lambda doc: doc.metadata["page"]
        )

        return docs

    def get_uploaded_documents(self):

        collection = Utils.mongo_client[
            Utils.DATABASE_NAME
        ][Utils.DOCUMENT_COLLECTION]

        return list(
            collection.find(
                {
                    "username": self.username
                },
                {
                    "_id": 0
                }
            )
        )