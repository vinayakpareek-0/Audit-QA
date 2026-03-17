import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from core.load_config import load_config
from core.retriever import BusinessRetriever
from core.memory import MemoryManager
from core.logger import AuditLogger
from core.intent import IntentClassifier
from core.extractor import LeadExtractor

load_dotenv()

class InferenceEngine:
    """Orchestrates RAG: Intent -> Hybrid Retrieval -> Memory -> Groq LLM Inference -> Lead Extraction -> Logging."""
    
    def __init__(self, session_id: str = "default-session", retriever: BusinessRetriever = None):
        self.config = load_config()
        self.retriever = retriever if retriever else BusinessRetriever()
        self.memory = MemoryManager(session_id)
        self.logger = AuditLogger()
        self.intent_clf = IntentClassifier()
        self.extractor = LeadExtractor()
        self.session_id = session_id
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        
        self.client = Groq(api_key=api_key)
        self.model = self.config['model']['name']
        self.temperature = self.config['model']['temperature']
        self.max_tokens = self.config['model']['max_tokens']

    def get_system_prompt(self, context: str, intent: str = "KNOWLEDGE") -> str:
        """Constructs a holistic system prompt with the 'Complete Picture' for the AI."""
        business_name = self.config.get('business', {}).get('name', 'Owlflow')
        
        prompt = f"""You are the official AI assistant for {business_name}. Your mission is to provide accurate, helpful knowledge based on company documents while facilitating human expertise when appropriate.

OWNER'S VISION & THE COMPLETE PICTURE:
1. You are a bridge between the customer and {business_name}'s human experts.
2. Build trust first by providing precise answers using ONLY the CONTEXT below.
3. Your ultimate goal is a fruitful business relationship. This means if a user is clearly interested, you should facilitate a way for them to connect with our experts.

CONTEXT FOR CURRENT SESSION:
{context}

OPERATIONAL FLOW & RULES:
- ACCURACY: Answer query based ONLY on context. If unknown, say: "I don't have that specific info yet, but I can check with our human team."
- CONCISION: Keep responses snappy (2-4 sentences). Don't overwhelm the user.
- FLOW: If the intent is '{intent}' and it's 'LEAD_GEN', it means the user is ready. Briefly provide the answer, then naturally bridge to engagement.
- TONE: Professional, efficient, and genuinely helpful. Don't be 'salesy', be an 'advocate' for the user's success.
"""
        return prompt.strip()

    def answer_question(self, query: str) -> tuple[str, dict]:
        """Runs the agentic RAG pipeline: Intent -> Retrieval -> LLM -> Lead Extraction."""
        metrics = {}
        
        # 1. Get History & Detect Intent
        history = self.memory.get_history()
        intent = self.intent_clf.classify(query, history)
        
        # 2. Retrieve Context
        start_retrieval = time.time()
        context = self.retriever.get_hybrid_context(query)
        metrics['retrieval_time'] = time.time() - start_retrieval
        
        # 3. Build Messages with Adaptive Prompt
        messages = [{"role": "system", "content": self.get_system_prompt(context, intent)}]
        messages.extend(history)
        messages.append({"role": "user", "content": query})
        
        # 4. Call Groq
        start_inference = time.time()
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        answer = response.choices[0].message.content
        metrics['inference_time'] = time.time() - start_inference
        metrics['total_time'] = metrics['retrieval_time'] + metrics['inference_time']
        
        # 5. Persist History
        self.memory.add_message("user", query)
        self.memory.add_message("assistant", answer)
        
        # 6. Background Tasks (Extract & Log) - DO NOT BLOCK THE UI
        def background_tasks():
            print(f"[DEBUG] Background tasks started. Intent: {intent}")
            try:
                self.logger.log_interaction(self.session_id, query, context, answer)
                
                # Extract leads regardless of "intent" classification to be safe
                updated_history = self.memory.get_history()
                print(f"[DEBUG] Extracting lead from history (length: {len(updated_history)})")
                lead_data = self.extractor.extract(updated_history)
                print(f"[DEBUG] Extracted lead data: {lead_data}")
                
                if lead_data:
                    self.logger.log_lead(self.session_id, lead_data)
                    print("[DEBUG] log_lead called.")
            except Exception as e:
                print(f"[DEBUG] Background task error: {e}")
        
        threading.Thread(target=background_tasks, daemon=True).start()
        
        return answer, metrics

if __name__ == "__main__":
    # Internal test
    try:
        engine = InferenceEngine()
        q = "What are the core services of Owlflow?"
        print(f"User: {q}")
        answer = engine.answer_question(q)
        print(f"Assistant: {answer}")
    except Exception as e:
        print(f"Error: {e}")
