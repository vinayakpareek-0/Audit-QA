import os
from supabase import create_client
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

class AuditLogger:
    """Logs RAG interactions to Supabase PostgreSQL for auditing."""
    
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") # Use Service Role for backend logging
        
        if not url or not key:
            print("[WARNING] Supabase credentials missing. Logging disabled.")
            self.client = None
        else:
            self.client = create_client(url, key)

    def log_interaction(self, session_id: str, query: str, context: str, response: str):
        """Inserts a new log entry into the 'audit_logs' table."""
        if not self.client:
            return

        data = {
            "session_id": session_id,
            "query": query,
            "context_summary": context[:500] + "..." if len(context) > 500 else context,
            "response": response,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            self.client.table("audit_logs").insert(data).execute()
        except Exception as e:
            print(f"[ERROR] Failed to log interaction to Supabase: {e}")

if __name__ == "__main__":
    # Internal test
    logger = AuditLogger()
    logger.log_interaction(
        session_id="test-log",
        query="Test query",
        context="Sample context",
        response="Sample response"
    )
