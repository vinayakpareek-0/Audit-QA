import os
import sys
import time
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

load_dotenv()

class InferenceEngine:
    """Orchestrates RAG: Hybrid Retrieval -> Memory -> Groq LLM Inference -> Logging."""
    
    def __init__(self, session_id: str = "default-session", retriever: BusinessRetriever = None):
        self.config = load_config()
        self.retriever = retriever if retriever else BusinessRetriever()
        self.memory = MemoryManager(session_id)
        self.logger = AuditLogger()
        self.session_id = session_id
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        
        self.client = Groq(api_key=api_key)
        self.model = self.config['model']['name']
        self.temperature = self.config['model']['temperature']
        self.max_tokens = self.config['model']['max_tokens']

    def get_system_prompt(self, context: str) -> str:
        """Constructs a grounding system prompt."""
        return f"""
        You are a specialized AI assistant for a small business. 
        Your goal is to provide accurate answers based ONLY on the provided context.
        Filter the provided context only based on asked query if context chunks are irrelevant to query.

        CORE CONTEXT:
        {context}

        INSTRUCTIONS:
        1. Answer based solely on the context provided.
        2. If the answer is not in the context, politely say you don't have that information.
        3. Keep the tone professional, helpful, and concise.
        """

    def answer_question(self, query: str) -> tuple[str, dict]:
        """Runs the full RAG pipeline with native message roles and logging. Returns (answer, metrics)."""
        metrics = {}
        
        # 1. Get History & Retrieve Context
        start_retrieval = time.time()
        history = self.memory.get_history()
        context = self.retriever.get_hybrid_context(query)
        metrics['retrieval_time'] = time.time() - start_retrieval
        
        # 2. Build Messages (Native roles)
        messages = [{"role": "system", "content": self.get_system_prompt(context)}]
        messages.extend(history)
        messages.append({"role": "user", "content": query})
        
        # 3. Call Groq
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
        
        # 4. Persist State
        self.memory.add_message("user", query)
        self.memory.add_message("assistant", answer)
        self.logger.log_interaction(self.session_id, query, context, answer)
        
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
