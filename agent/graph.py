from typing import TypedDict
from agent.models import CompareDocuments
from agent.prompts import SUMMARY_PROMPT
from agent.summarizer import GROUP_SIZE, summarize_document
from agent.tools import compare_tool, extract_tool, qa_tool, _format_context,summary_tool
from langgraph.graph import StateGraph, START, END
from agent.router import Route
import utils as Utils

class GraphState(TypedDict):
    question: str
    chat_history: list

    tool: str

    docs: list
    scores: list

    answer: str


class SmartDocsGraph:

    def __init__(self, context):

        self.context = context

        workflow = StateGraph(GraphState)
        workflow.add_node("router", self.router_node)
        workflow.add_node("qa", self.qa_node)
        workflow.add_node("save_history", self.save_history_node)
        workflow.add_node("summary",self.summary_node)

        workflow.add_node("compare",self.compare_node)
        workflow.add_node("search_all",self.qa_node)
        workflow.add_node("extract",self.extract_node)

        workflow.add_edge(START, "router")
        workflow.add_edge("qa", "save_history")

        workflow.add_edge("summary","save_history")

        workflow.add_edge("compare","save_history")

        workflow.add_edge("extract","save_history")
        workflow.add_edge("search_all","save_history")
        workflow.add_edge("save_history", END)

        self.router_llm = self.context.llm.with_structured_output(Route)

        workflow.add_conditional_edges(
            "router",
            self.route,
            {
                "qa": "qa",
                "summary": "summary",
                "compare": "compare",
                "extract": "extract",
                "search_all": "search_all"
            },
        )

        self.graph = workflow.compile()

    def router_node(self, state):

        response = self.router_llm.invoke(
                f"""
            You are the routing system for SmartDocs.

            Select exactly one tool.

            Available tools:

            1. qa
            - Use when the user asks a question whose answer should be found in the document.

            Examples:
            - What is Article 5?
            - Explain this section.
            - Who is the author?

            2. summary
            - Use when the user wants a summary.

            Examples:
            - Summarize this document.
            - Give me a short summary.
            - TLDR.

            compare
            - Compare two uploaded documents.

            Examples:
            - Compare resume.pdf and smartdocs.pdf
            - Compare report1.pdf with report2.pdf
            - Difference between contract.pdf and invoice.pdf
            -Compare this documents

            If the user specifies filenames,
            Return the filenames in the documents field.

            4. search_all
            - Search across ALL uploaded documents instead of only the currently selected document.

            Use this when the user wants to:
            - Find which document contains specific information.
            - Search for a keyword across all uploaded PDFs.
            - Locate a topic without knowing which document it belongs to.

            Examples:
            - Search all documents for MongoDB.
            - Which document mentions LangGraph?
            - Find every PDF that discusses GDPR.
            - Search across all my uploaded documents for Article 5.
            - Which document contains information about vector databases?
            - Find documents mentioning machine learning.

            Do NOT use this tool if the user is asking about the currently selected document.

            5. extract
            - Use when the user wants structured information.

            Examples:
            - Extract all dates.
            - Extract all organizations.
            - Extract all emails.

            User Query:

            {state["question"]}
            """
            )

        print("=" * 50)
        print("Selected Tool :", response.tool)
        print("=" * 50)

        return {
            "tool": response.tool
        }

    def route(self, state):
        return state["tool"]

    def qa_node(self, state):

        search_kwargs = {
            "k": Utils.DEFAULT_TOP_K
        }

        pre_filter = {}

        if state['tool']=='qa' and self.context.document_id:
            pre_filter["document_id"] = self.context.document_id

        if self.context.username:
            pre_filter["username"] = self.context.username

        if pre_filter:
            search_kwargs["pre_filter"] = pre_filter

        results = self.context.vector_store.similarity_search_with_score(
            state["question"],
            **search_kwargs
        )

        docs = [doc for doc, _ in results]
        scores = [score for _, score in results]

        answer = qa_tool(
            self.context.llm,
            state["question"],
            _format_context(docs),
            state["chat_history"],
        )

        seen = set()
        sources = []

        for doc, score in zip(docs, scores):

            filename = doc.metadata.get("filename", "Unknown")
            page = doc.metadata.get("page", 0) + 1

            key = (filename, page)

            if key in seen:
                continue

            seen.add(key)

            sources.append(
                f"• {filename} | Page {page} | Similarity: {score * 100:.2f}%"
            )

        if sources:
            answer += "\n\n📚 Sources\n"
            answer += "\n".join(sources)

        return {
            "answer": answer,
            "docs": docs,
            "scores": scores,
        }
    
    def save_history_node(self, state):

        self.context.memory.add_user_message(
            state["question"]
        )

        self.context.memory.add_ai_message(
            state["answer"]
        )

        return {}

    def invoke(self, state):
        return self.graph.invoke(state)
    
    
    def summary_node(self, state):

        docs = self.context.get_document()

        answer = summarize_document(
            self.context.llm,
            docs,
        )

        return {
            "answer": answer,
            "docs": docs,
            "scores": [],
        }

    def compare_node(self, state):

        uploaded = self.context.get_uploaded_documents()
        available = "\n".join(
            doc["filename"]
            for doc in uploaded
        )

        selector = self.context.llm.with_structured_output(CompareDocuments)

        selection = selector.invoke(
            f"""
        Available documents

        {available}

        User request

        {state["question"]}

        Return exactly two filenames.
        """
        )

        print(selection)

        docs1 = self.context.get_document(
            selection.documents[0]
        )

        docs2 = self.context.get_document(
            selection.documents[1]
        )

        answer = compare_tool(
            self.context.llm,
            _format_context(docs1),
            _format_context(docs2),
        )

        return {
            "answer": answer,
            "docs": docs1 + docs2,
            "scores": [],
        }



    def extract_node(self, state):

        docs = self.context.get_document()

        answer = extract_tool(
            self.context.llm,
            state["question"],
            _format_context(docs),
        )

        return {
            "answer": answer,
            "docs": docs,
            "scores": [],
        }