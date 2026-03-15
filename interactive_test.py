from core.inference import InferenceEngine

root_path = Path(__file__).resolve().parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

def main():
    try:
        engine = InferenceEngine()
        print("QA Session started. Type 'exit' to quit.")
        
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
