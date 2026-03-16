import os
from groq import Groq
from dotenv import load_dotenv
from core.load_config import load_config

load_dotenv()

class IntentClassifier:
    """Classifies user intent to determine if it's time to pivot to lead generation."""
    
    def __init__(self):
        self.config = load_config()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = self.config['model'].get('small_model', "llama-3.1-8b-instant")
        
    def classify(self, query: str, history: list) -> str:
        """
        Classifies the query and history into a state: 'KNOWLEDGE' or 'LEAD_GEN'.
        KNOWLEDGE: Standard Q&A.
        LEAD_GEN: User shows high interest, asks about pricing, or provides contact info.
        """
        history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history[-3:]])
        
        prompt = f"""
        Analyze the following chat interaction. 
        Your goal is to decide if the user is showing "Buying Intent" or generic interest in the company's services.

        RECENT HISTORY:
        {history_text}

        CURRENT QUERY:
        "{query}"

        CLASSIFICATION RULES:
        1. If the user asks about price, hiring, contact details, procedure to join, or says "I'm interested", return "LEAD_GEN".
        2. If the user provides personal info (name, email, phone) or asks "how do I start", return "LEAD_GEN".
        3. If the user asks for a demo, scheduling, or meeting, return "LEAD_GEN".
        4. Otherwise, return "KNOWLEDGE".

        Return ONLY the word: KNOWLEDGE or LEAD_GEN.
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0,
                max_tokens=10
            )
            intent = response.choices[0].message.content.strip().upper()
            return intent if intent in ["KNOWLEDGE", "LEAD_GEN"] else "KNOWLEDGE"
        except Exception as e:
            print(f"[ERROR] Intent classification failed: {e}")
            return "KNOWLEDGE"

if __name__ == "__main__":
    classifier = IntentClassifier()
    # Test cases
    print(f"Test 1: {classifier.classify('How much does it cost?', [])}") # Expected: LEAD_GEN
    print(f"Test 2: {classifier.classify('Who is the CEO?', [])}")     # Expected: KNOWLEDGE
