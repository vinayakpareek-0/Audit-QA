import os
import sys
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

# Add project root to sys.path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from core.load_config import load_config
from core.retriever import BusinessRetriever

load_dotenv()

class InferenceEngine:
    """Orchestrates RAG: Hybrid Retrieval -> Groq LLM Inference."""
    
    def __init__(self):
        self.config = load_config()
        self.retriever = BusinessRetriever()
        
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

        CONTEXT:
        {context}

        INSTRUCTIONS:
        1. Answer based solely on the context above.
        2. If the answer is not in the context, politely say you don't have that information.
        3. Keep the tone professional, helpful, and concise.
        4. Do not mention the context or internal retrieval process in your answer.
        """

    def answer_question(self, query: str) -> str:
        """Runs the full RAG pipeline to answer a query."""
        # 1. Retrieve Context
        context = self.retriever.get_hybrid_context(query)
        
        # 2. Build Messages
        messages = [
            {"role": "system", "content": self.get_system_prompt(context)},
            {"role": "user", "content": query}
        ]
        
        # 3. Call Groq
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content

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
