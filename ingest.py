import os
import json
import sys
from pathlib import Path
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

root_path = Path(__file__).resolve().parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from core.load_config import load_config

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content: text += content + "\n"
    return text

def ingest_data():
    config = load_config()
    client = chromadb.PersistentClient(path=config['paths']['vectorstore_dir'])
    col_name = config['retrieval'].get('collection_name', 'business_kb')
    collection = client.get_or_create_collection(name=col_name)
    model = SentenceTransformer(config['embeddings']['model_name'])

    documents, metadatas, ids = [], [], []

    if os.path.exists(config['data']['pdf_path']):
        text = load_pdf(config['data']['pdf_path'])
        documents.append(text)
        metadatas.append({"source": "pdf"})
        ids.append("pdf_main")

    if os.path.exists(config['data']['kb_path']):
        with open(config['data']['kb_path'], 'r') as f:
            data = json.load(f)
            for i, item in enumerate(data):
                documents.append(item['text'])
                metadatas.append({"source": "kb_json"})
                ids.append(f"kb_{i}")

    if os.path.exists(config['data']['qa_path']):
        with open(config['data']['qa_path'], 'r') as f:
            data = json.load(f)
            for i, item in enumerate(data):
                documents.append(f"Question: {item['question']}\nAnswer: {item['answer']}")
                metadatas.append({"source": "qa_json"})
                ids.append(f"qa_{i}")

    if documents:
        embeddings = model.encode(documents).tolist()
        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

if __name__ == "__main__":
    ingest_data()

if __name__ == "__main__":
    ingest_data()
