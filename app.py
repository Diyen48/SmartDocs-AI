import os
import streamlit as ST
import agent as Agent
import utils as Utils
import embed

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def create_chat(chat_id: str):
    chat = ST.container()
    with ST.sidebar:
        ST.divider()
        ST.subheader("Settings")
        show_sources = ST.checkbox(
            "Show Sources",
            value=True
        )
        show_chunks = ST.checkbox(
            "Show Retrieved Chunks",
            value=False
        )
    document_id = ST.session_state.get("document_id")
    if document_id is None:
        ST.info("Please select a document.")
        return
    if "messages" not in ST.session_state:
        ST.session_state.messages = {}

    if document_id not in ST.session_state.messages:
        ST.session_state.messages[document_id] = []

    # Replay stored messages for this chat session
    for msg in ST.session_state.messages[document_id]:
        chat.chat_message(msg["role"]).write(msg["content"])


    if ("news_chat" not in ST.session_state
        or ST.session_state.get("chat_document") != document_id):
        ST.session_state.news_chat = Agent.NewsChat(
            chat_id,
            document_id,
            ST.session_state["username"]
        )


    ST.session_state.chat_document = document_id

    news_chat = ST.session_state.news_chat

    if prompt := ST.chat_input(placeholder="Ask a question about your document...", key=chat_id):
        chat.chat_message("user").write(prompt)
        with ST.spinner("Thinking..."):
            response = news_chat.ask(prompt)
        chat.chat_message("assistant").write(response['answer'])

        sources = response["documents"]
        scores = response["scores"]

        if show_sources and sources:
            with chat.expander("📚 Sources"):
                shown = set()
                for doc,score in zip(sources,scores):
                    page = doc.metadata.get("page", 0) + 1
                    filename = doc.metadata.get("filename", "Unknown")
                    key = (filename, page)
                    if key in shown:
                        continue
                    shown.add(key)
                    ST.markdown(f"### 📄 {filename}")
                    ST.write(f"**Page:** {page}")
                    ST.write(f"**Similarity Score:** `{score*100:.2f}`%")
                    ST.divider()
        ST.session_state.messages[document_id].append({"role": "user","content": prompt})
        ST.session_state.messages[document_id].append({"role": "assistant", "content": response['answer']})

        if show_chunks:
            with chat.expander("📄 Retrieved Chunks"):
                for i, doc in enumerate(response["retrieved_chunks"], start=1):
                    ST.markdown(f"### Chunk {i}")
                    ST.markdown(
                        f"**Document:** {doc.metadata.get('filename','Unknown')}"
                    )

                    ST.markdown(
                        f"**Page:** {doc.metadata.get('page',0)+1}"
                    )

                    ST.write(doc.page_content)
                    ST.divider()


def upload_pdf():

    uploaded_file = ST.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file is None:
        return

    # Prevent embedding the same upload again on rerun
    if (
        ST.session_state.get("last_uploaded_file") == uploaded_file.name
        and ST.session_state.get("document_id") is not None
    ):
        return

    save_path = os.path.join(
        UPLOAD_FOLDER,
        uploaded_file.name
    )

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    ST.success(f"{uploaded_file.name} uploaded successfully!")

    with ST.spinner("Embedding document..."):

        document_id = embed.embed_and_store_from_path(
            save_path,
            uploaded_file.name,
            ST.session_state["username"]
        )

    ST.session_state["document_id"] = document_id
    ST.session_state["selected_document"] = uploaded_file.name
    ST.session_state["last_uploaded_file"] = uploaded_file.name

    ST.success("Document embedded successfully!")



def main():
    ST.set_page_config(page_title="AI Doc Assistant", page_icon="📄")
    ST.title("📄 SmartDocs AI")
    ST.caption("Chat with your PDF documents using Local AI + RAG")
    if "username" not in ST.session_state:

        ST.subheader("👤 Enter Username")

        username = ST.text_input("Username")

        if ST.button("Continue"):

            if username.strip():

                ST.session_state["username"] = username.strip()

                ST.rerun()

        ST.stop()
    with ST.sidebar:
        if ST.button("Logout"):
            ST.session_state.clear()
            ST.rerun()
        ST.success(f"👤 {ST.session_state['username']}")
        ST.header("📄 Documents")
        ST.subheader("⬆ Upload PDF")
        upload_pdf()
        ST.divider()
        documents = embed.get_uploaded_documents(ST.session_state["username"])
        if documents:
            ST.subheader("📄 My Documents")
            selected = ST.selectbox(
                "Choose document:",
                documents,
                format_func=lambda x: x["filename"]
            )
            ST.info(f"Current document:\n\n{selected['filename']}")
            ST.session_state["document_id"] = selected["document_id"]
            ST.session_state["selected_document"] = selected["filename"]

            # Reset upload tracking
            ST.session_state["last_uploaded_file"] = None
        
    if  ST.session_state.get("document_id"):
        create_chat("chat1")
    else:
        ST.info("📄 Please upload or select a document to start chatting.")


if __name__ == "__main__":
    main()