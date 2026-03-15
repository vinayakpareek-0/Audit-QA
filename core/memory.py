import os
import json
from upstash_redis import Redis
from dotenv import load_dotenv

load_dotenv()

class MemoryManager:
    """Handles persistent conversation history using Upstash Redis."""
    
    def __init__(self, session_id: str):
        self.session_id = f"chat_history:{session_id}"
        self.redis = Redis(
            url=os.getenv("UPSTASH_REDIS_REST_URL"),
            token=os.getenv("UPSTASH_REDIS_REST_TOKEN")
        )
        self.history_limit = 10 # Only keep last 5 turns (10 messages)

    def add_message(self, role: str, content: str):
        """Appends a message to the session history."""
        message = json.dumps({"role": role, "content": content})
        self.redis.rpush(self.session_id, message)
        
        # Trim history to keep it fast
        self.redis.ltrim(self.session_id, -self.history_limit, -1)

    def get_history(self):
        """Retrieves and parses the conversation history as a list of dicts."""
        history = self.redis.lrange(self.session_id, 0, -1)
        return [json.loads(msg) for msg in history] if history else []

    def clear_history(self):
        """Wipes the session history."""
        self.redis.delete(self.session_id)

if __name__ == "__main__":
    # Internal test
    test_id = "test-session-123"
    mem = MemoryManager(test_id)
    
    print("Testing Memory write...")
    mem.add_message("user", "Chai kitne ki hai bhaiya?")
    mem.add_message("assistant", "15 ki hai 20 ki hogi ab")
    
    print("Testing Memory read...")
    history = mem.get_history()
    for msg in history:
        print(f"{msg['role'].upper()}: {msg['content']}")
    
    # mem.clear_history()
