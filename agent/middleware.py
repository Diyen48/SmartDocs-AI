from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage
import utils as Utils

class SummaryMiddleware:

    def __init__(self, context):

        self.context = context
        self.MAX_MESSAGES = Utils.MAX_HISTORY_MESSAGES
        self.memory = context.memory
        self.llm = context.llm

    def get_history(self) -> list[BaseMessage]:

        messages = self.memory.load_messages()

        if len(messages) > self.MAX_MESSAGES:
            self._compress_history(messages)
            messages = self.memory.load_messages()

        return messages

    def _compress_history(self, messages):

        conversation = []

        for message in messages:

            if isinstance(message, HumanMessage):
                role = "User"

            elif isinstance(message, AIMessage):
                role = "Assistant"

            else:
                role = "System"

            conversation.append(
                f"{role}: {message.content}"
            )

        conversation = "\n".join(conversation)

        summary = self.llm.invoke(
            f"""
        Summarize this conversation.

        Keep only:
        - Important facts
        - Previous discussion
        - User preferences

        Conversation:

        {conversation}
        """
        ).content

        self.memory.clear()

        self.memory.add_ai_message(
            f"Conversation Summary:\n\n{summary}"
        )

        