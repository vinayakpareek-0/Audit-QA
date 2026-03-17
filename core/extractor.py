import os
import json
from groq import Groq
from dotenv import load_dotenv
from core.load_config import load_config

load_dotenv()

class LeadExtractor:
    """Extracts structured lead information from conversation history."""
    
    def __init__(self):
        self.config = load_config()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = self.config['model'].get('name', "llama-3.3-70b-versatile")
        
    def extract(self, history: list) -> dict:
        """Extracts Name, Email, Phone, and Use-case from history."""
        if not history:
            return {"full_name": None, "email": None, "phone": None, "use_case": None}

        history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history])
        
        prompt = f"""
        Analyze the full conversation history to identify a potential business partner.
        Your goal is to extract a 'Lead Profile' for the Sales Team.
        
        HISTORY:
        {history_text}

        EXTRACTION RULES:
        1. "full_name": Extract the user's name if shared.
        2. "email": Extract valid emails.
        3. "phone": Extract phone numbers.
        4. "use_case": This is the most important. Don't just extract a single word. Synthesize the user's problem, industry, and why they are interested in {self.config.get('business', {}).get('name', 'the business')}.

        RETURN A JSON OBJECT WITH THESE KEYS:
        "full_name", "email", "phone", "use_case" (all keys must be present).
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            # Ensure all keys exist
            for key in ["full_name", "email", "phone", "use_case"]:
                if key not in data:
                    data[key] = None
            return data
        except Exception as e:
            print(f"[ERROR] Lead extraction failed: {e}")
            return {"full_name": None, "email": None, "phone": None, "use_case": None}

if __name__ == "__main__":
    extractor = LeadExtractor()
    test_history = [
        {"role": "user", "content": "My name is Vinay and I want to automate my tea shop."},
        {"role": "assistant", "content": "That sounds great, Vinay! How can I help?"},
        {"role": "user", "content": "Can you contact me at vpj@example.com?"}
    ]
    print(f"Extracted: {extractor.extract(test_history)}")
