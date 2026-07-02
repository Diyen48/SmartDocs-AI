from langchain_mongodb import MongoDBChatMessageHistory
import utils as Utils


class ConversationMemory:

    def __init__(self, session_id: str):
        self.history = MongoDBChatMessageHistory(
            connection_string=Utils.MONGODB_URI,
            session_id=session_id,
            database_name=Utils.DATABASE_NAME,
            collection_name=Utils.CHAT_COLLECTION,
        )

    def load_messages(self):
        return self.history.messages

    def add_user_message(self, message: str):
        self.history.add_user_message(message)

    def add_ai_message(self, message: str):
        self.history.add_ai_message(message)