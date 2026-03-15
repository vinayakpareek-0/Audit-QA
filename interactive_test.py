import sys
from pathlib import Path
from core.retriever import BusinessRetriever

root_path = Path(__file__).resolve().parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

def main():
    retriever = BusinessRetriever()
    print("Session started. Type 'exit' to quit.")
    
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in ['exit', 'quit']:
            break
        if not query:
            continue
            
        context = retriever.get_hybrid_context(query)
        print("\nContext:\n", context)

if __name__ == "__main__":
    main()
