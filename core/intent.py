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
        
        prompt = f"""History: {history_text}\nQuery: {query}\nRules: LEAD_GEN if user shows interest/price/hiring/contact info. Else KNOWLEDGE. Output 1 word."""
        
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
