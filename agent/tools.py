from agent.prompts import COMPARE_PROMPT, EXTRACT_PROMPT, QA_PROMPT,SUMMARY_PROMPT

def _format_context(docs):

    formatted = []

    for doc in docs:

        page = doc.metadata.get("page", 0) + 1

        filename = doc.metadata.get("filename", "Unknown")

        formatted.append(
            f"""
        ========== SOURCE ==========

        Filename: {filename}
        Page: {page}

        Content:
        {doc.page_content}
        """
        )

    return "\n\n-----------------\n\n".join(formatted)

def qa_tool(llm,question,context,chat_history,):

    messages = QA_PROMPT.invoke(
        {
            "question": question,
            "context": context,
            "chat_history": chat_history,
        }
    )

    response = llm.invoke(messages)

    return response.content

def summary_tool(llm,context):

    messages = SUMMARY_PROMPT.invoke(
        {
            "context": context,
        }
    )

    response = llm.invoke(messages)

    return response.content

def extract_tool(llm,question,context):

    messages = EXTRACT_PROMPT.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    response = llm.invoke(messages)

    return response.content

def compare_tool(llm,document1,document2):

    messages = COMPARE_PROMPT.invoke(
        {
            "document1": document1,
            "document2": document2,
        }
    )

    response = llm.invoke(messages)

    return response.content