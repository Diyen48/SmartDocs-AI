import json as JS
import os as OS
from dotenv import load_dotenv
from pymongo import MongoClient
load_dotenv()

ARTICLES_FILE    = 'articles.json'
ARTICLES_FOLDER  = 'articles'
DATA_FOLDER      = 'data'
EUROPEAN_ACT_URL = 'https://www.europarl.europa.eu/doceo/document/TA-9-2024-0138_EN.pdf'
DOCUMENT_COLLECTION = "documents"
# MongoDB Atlas
MONGODB_URI       = OS.getenv('MONGODB_URI', '')
DATABASE_NAME     = OS.getenv('DATABASE_NAME', 'legalai')
COLLECTION_NAME   = OS.getenv('COLLECTION_NAME', 'legal_documents')
CHAT_COLLECTION   = OS.getenv('CHAT_COLLECTION', 'chat_history')
VECTOR_INDEX_NAME = OS.getenv('VECTOR_INDEX_NAME', 'vector_index')
DEFAULT_TOP_K     = int(OS.getenv('DEFAULT_TOP_K', '4'))
MAX_HISTORY_MESSAGES = 30
GROQ_API_KEY = OS.getenv('GROQ_API_KEY')
mongo_client = MongoClient(MONGODB_URI)
def load_articles(file_name) -> list:
    result = []
    if OS.path.exists(file_name):
        with open(file_name, 'r') as f:
            try:
                result = JS.load(f)
            except JS.JSONDecodeError:
                print("File is not valid JSON. Returning empty list.")
    else:
        with open(file_name, 'w') as f:
            JS.dump([], f)
        print(f"Created '{file_name}'.")
        if not OS.path.exists(ARTICLES_FOLDER):
            OS.mkdir(ARTICLES_FOLDER)
    return result

def save_articles(file_name, data):
    try:
        with open(file_name, 'w') as f:
            JS.dump(data, f, indent=4)
        print(f"Saved to '{file_name}'.")
    except Exception as e:
        print(f"Error saving: {e}")

def save_article_content(file_name, content):
    try:
        with open(file_name, 'w') as f:
            f.write(content)
        print(f"Written to '{file_name}'.")
    except Exception as e:
        print(f"Error writing: {e}")

def load_article_content(file_name):
    try:
        with open(file_name, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading '{file_name}': {e}")
        return ''