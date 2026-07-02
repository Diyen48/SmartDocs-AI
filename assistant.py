from agent.context import SmartDocsContext
from agent.graph import SmartDocsGraph
from agent.middleware import SummaryMiddleware


class NewsChat:
    def __init__(self, session_id: str, document_id: str = None,username=None):
        print("Creating NewsChat...")
        self.session_id = f"{username}_{document_id}"
        self.username=username
        self.document = document_id
        print("Using document:", document_id)
        self.context = SmartDocsContext(
            session_id=self.session_id,
            username=self.username,
            document_id=self.document,
        )
        self.middleware = SummaryMiddleware(self.context)
        self.graph = SmartDocsGraph(self.context)  
    

    def ask(self, question):

        chat_history = self.middleware.get_history()

        state = self.graph.invoke(
            {
                "question": question,
                "chat_history": chat_history,
            }
        )

        return {
            "answer": state["answer"],
            "documents": state["docs"],
            "scores": state["scores"],
            "retrieved_chunks": state["docs"],
        }