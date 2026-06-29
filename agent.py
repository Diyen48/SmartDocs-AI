from pymongo import MongoClient
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch, MongoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
import utils as Utils


def _format_docs(docs):

    formatted = []

    for doc in docs:

        page = doc.metadata.get("page", 0) + 1

        filename = doc.metadata.get("filename", "Unknown")

        formatted.append(
            f"""
                Source: {filename}
                Page: {page}

                {doc.page_content}
                """
            )

    return "\n\n-----------------\n\n".join(formatted)


class NewsChat:
    def __init__(self, session_id: str, document_id: str = None):
        print("Creating NewsChat...")
        if document_id:
            self.session_id = f"{session_id}_{document_id}"
        else:
            self.session_id = session_id

        self.document = document_id
        print("Using document:", document_id)
        llm        = ChatOllama(model="llama3.2:3b", temperature=0)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # ── Vector store + retriever ──────────────────────────────────────
        client     = Utils.mongo_client
        collection = client[Utils.DATABASE_NAME][Utils.COLLECTION_NAME]

        self.vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name=Utils.VECTOR_INDEX_NAME,
            text_key="text",
            embedding_key="embedding",
        )

        

        # ── Load chat history from MongoDB ────────────────────────────────
        self.history = MongoDBChatMessageHistory(
            connection_string=Utils.MONGODB_URI,
            session_id=self.session_id,
            database_name=Utils.DATABASE_NAME,
            collection_name=Utils.CHAT_COLLECTION,
        )

        # ── Prompt: answer the question using retrieved context ───────────
        qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
    You are SmartDocs AI, an intelligent document assistant.

    Your job is to answer the user's question ONLY using the retrieved document context provided below.

    Rules:
    1. Never use your own knowledge or make up information.
    2. If the answer is not present in the retrieved context, reply:
    "I couldn't find this information in the uploaded document."
    3. Answer in clear, professional English.
    4. Keep the answer concise (3-6 sentences unless more detail is required).
    5. If the retrieved context contains page information, naturally mention the page number in your answer when appropriate.
    6. Do not mention the retrieval process, embeddings, vector database, or AI models.
    7. If multiple retrieved chunks contain relevant information, combine them into one coherent answer.
    8. If the question is ambiguous, ask a clarifying question instead of guessing.

    Retrieved Context:
    -----------------
    {context}
    """
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
])

        # ── RAG chain (LCEL) ──────────────────────────────────────────────
        # Step 1: retrieve docs using the question
        # Step 2: format docs into context string
        # Step 3: fill prompt and call LLM

        self.qa_chain = (
            qa_prompt
            | llm
            | StrOutputParser()
        )

    def ask(self, question: str):

        # Load chat history
        chat_history = self.history.messages

        search_kwargs = {
            "k": Utils.DEFAULT_TOP_K
        }

        if self.document:
            search_kwargs["pre_filter"] = {
                "document_id": self.document
            }

        results = self.vector_store.similarity_search_with_score(
            question,
            **search_kwargs
        )

        docs = [doc for doc, score in results]

        scores = [score for doc, score in results]

        context = _format_docs(docs)

        # Generate answer
        answer = self.qa_chain.invoke({
            "question": question,
            "context": context,
            "chat_history": chat_history,
        })

        # Save history
        self.history.add_user_message(question)
        self.history.add_ai_message(answer)

        # Return structured response
        return {
            "answer": answer,
            "documents": docs,
            "scores": scores,
            "retrieved_chunks": docs,
        }