from core.inference import InferenceEngine
from pathlib import Path
import os
import sys

root_path = Path(__file__).resolve().parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

def main():
    try:
        session_id = input("Enter Session ID (or press Enter for default): ").strip() or "default-session"
        engine = InferenceEngine(session_id=session_id)
        print(f"QA Session '{session_id}' started. Type 'exit' to quit.")
        
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            if not query:
                continue
                
            response = engine.answer_question(query)
            print("\nAnswer:\n", response)
    except Exception as e:
        print(f"Error initializing engine: {e}")

if __name__ == "__main__":
    main()
