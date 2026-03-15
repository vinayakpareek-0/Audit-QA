import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
from datetime import datetime, timezone

# Add project root to sys.path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

# Load .env from project root
load_dotenv(root_path / ".env")

class AuditLogger:
    """Logs RAG interactions to Supabase PostgreSQL for auditing."""
    
    def __init__(self):
        # Look for both SUPABASE_URL and SUPABASE_API_URL
        url = os.getenv("SUPABASE_URL") or os.getenv("SUPABASE_API_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not url or not key:
            # Try to construct URL from DB_URL if missing
            db_url = os.getenv("SUPABASE_DB_URL")
            if db_url and "db." in db_url:
                ref_id = db_url.split("db.")[1].split(".")[0]
                url = f"https://{ref_id}.supabase.co"
            
        if not url or not key:
            print("[WARNING] Supabase API URL or Service Key missing. Logging disabled.")
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
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            self.client.table("audit_logs").insert(data).execute()
        except Exception as e:
            print(f"[ERROR] Failed to log interaction to Supabase: {e}")

    def log_lead(self, session_id: str, lead_data: dict):
        """Inserts a lead in the 'leads' table if meaningful data exists."""
        if not self.client:
            return

        # Check if we have at least SOME useful info
        has_info = any([
            lead_data.get("full_name"),
            lead_data.get("email"),
            lead_data.get("phone")
        ])
        
        if not has_info:
            print("[DEBUG] Lead extraction returned no personal info. Skipping log.")
            return

        data = {
            "session_id": session_id,
            "full_name": lead_data.get("full_name"),
            "email": lead_data.get("email"),
            "phone": lead_data.get("phone"),
            "use_case": lead_data.get("use_case"),
            "status": "New"
        }
        
        try:
            self.client.table("leads").insert(data).execute()
        except Exception as e:
            print(f"[ERROR] Failed to log lead to Supabase: {e}")

if __name__ == "__main__":
    # Internal test
    logger = AuditLogger()
    logger.log_interaction(
        session_id="test-log",
        query="Test query",
        context="Sample context",
        response="Sample response"
    )
