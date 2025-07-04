import sqlite3
import logging
import os
import threading
from typing import List, Dict, Optional, Any
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
DB_PATH = "./chroma_db"  # Directory to store database
COLLECTION_NAME = "agentk_memories"
EMBEDDING_MODEL_NAME = "bge-small-en"

# Thread-local storage for SQLite connections
thread_local = threading.local()


# --- MemoryManager Class ---
class MemoryManager:
    def __init__(self):
        # Ensure DB directory exists
        os.makedirs(DB_PATH, exist_ok=True)

        # Initialize the database file and create table in the main thread
        self._init_db()

    def _init_db(self):
        """Initialize the database file and create table"""
        try:
            # Create the database file if it doesn't exist
            db_path = os.path.join(DB_PATH, "memory.db")
            if not os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                self._create_table(conn)
                conn.close()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}", exc_info=True)
            raise RuntimeError(f"Database initialization failed: {e}") from e

    def _get_connection(self):
        """Get a thread-local SQLite connection"""
        if not hasattr(thread_local, "connection"):
            thread_local.connection = sqlite3.connect(
                os.path.join(DB_PATH, "memory.db")
            )
            # Enable foreign keys and WAL mode for better concurrency
            thread_local.connection.execute("PRAGMA foreign_keys = ON")
            thread_local.connection.execute("PRAGMA journal_mode = WAL")
        return thread_local.connection

    def _create_table(self, conn):
        """Create the memory table if it doesn't exist"""
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    memory_type TEXT,
                    agent_name TEXT,
                    task_info TEXT,
                    importance INTEGER DEFAULT 5,
                    timestamp TEXT
                )
            """
            )
            conn.commit()
            logger.info("Memory table created or already exists")
        except Exception as e:
            logger.error(f"Failed to create memory table: {e}", exc_info=True)
            raise

    def add_memory(
        self,
        key: str,
        value: str,
        memory_type: str = None,
        agent_name: str = None,
        task_info: str = None,
        importance: int = 5,
    ) -> bool:
        """
        Adds a memory to the database.

        Args:
            key: The unique key for this memory.
            value: The content of the memory.
            memory_type: Type of memory (e.g., 'goal', 'plan', 'solution').
            agent_name: (Optional) Name of the agent associated with this memory.
            task_info: (Optional) Description of the task related to this memory.
            importance: An integer score (1-10) indicating memory importance. Defaults to 5.

        Returns:
            True if successful, False otherwise.
        """
        if not key or not value:
            logger.warning("Attempted to add memory with empty key or value")
            return False

        try:
            conn = self._get_connection()
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            conn.execute(
                """INSERT OR REPLACE INTO memory 
                   (key, value, memory_type, agent_name, task_info, importance, timestamp) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (key, value, memory_type, agent_name, task_info, importance, timestamp),
            )
            conn.commit()
            logger.info(
                f"Added memory with key: {key}, type: {memory_type}, importance: {importance}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add memory: {e}", exc_info=True)
            return False

    def retrieve_memory(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific memory by its key.

        Args:
            key: The unique key of the memory to retrieve.

        Returns:
            Dict containing memory data or None if not found.
        """
        try:
            conn = self._get_connection()
            cur = conn.execute(
                """SELECT key, value, memory_type, agent_name, task_info, importance, timestamp 
                   FROM memory WHERE key = ?""",
                (key,),
            )
            result = cur.fetchone()

            if result:
                return {
                    "key": result[0],
                    "value": result[1],
                    "memory_type": result[2],
                    "agent_name": result[3],
                    "task_info": result[4],
                    "importance": result[5],
                    "timestamp": result[6],
                }
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve memory: {e}", exc_info=True)
            return None

    def retrieve_memories(
        self,
        memory_type: str = None,
        agent_name: str = None,
        min_importance: int = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories based on filter criteria.

        Args:
            memory_type: Filter by memory type.
            agent_name: Filter by agent name.
            min_importance: Minimum importance score.
            limit: Maximum number of memories to retrieve.

        Returns:
            List of memory dictionaries.
        """
        try:
            conn = self._get_connection()
            query = "SELECT key, value, memory_type, agent_name, task_info, importance, timestamp FROM memory WHERE 1=1"
            params = []

            if memory_type:
                query += " AND memory_type = ?"
                params.append(memory_type)

            if agent_name:
                query += " AND agent_name = ?"
                params.append(agent_name)

            if min_importance is not None:
                query += " AND importance >= ?"
                params.append(min_importance)

            query += " ORDER BY importance DESC LIMIT ?"
            params.append(limit)

            cur = conn.execute(query, tuple(params))
            results = cur.fetchall()

            memories = []
            for row in results:
                memories.append(
                    {
                        "key": row[0],
                        "value": row[1],
                        "memory_type": row[2],
                        "agent_name": row[3],
                        "task_info": row[4],
                        "importance": row[5],
                        "timestamp": row[6],
                    }
                )

            return memories
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}", exc_info=True)
            return []

    def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Simple text search in memory values.

        Args:
            query: Text to search for in memory values.
            limit: Maximum number of results to return.

        Returns:
            List of memory dictionaries matching the search.
        """
        try:
            conn = self._get_connection()
            cur = conn.execute(
                """SELECT key, value, memory_type, agent_name, task_info, importance, timestamp 
                   FROM memory WHERE value LIKE ? ORDER BY importance DESC LIMIT ?""",
                (f"%{query}%", limit),
            )
            results = cur.fetchall()

            memories = []
            for row in results:
                memories.append(
                    {
                        "key": row[0],
                        "value": row[1],
                        "memory_type": row[2],
                        "agent_name": row[3],
                        "task_info": row[4],
                        "importance": row[5],
                        "timestamp": row[6],
                    }
                )

            return memories
        except Exception as e:
            logger.error(f"Failed to search memories: {e}", exc_info=True)
            return []

    def delete_memory(self, key: str) -> bool:
        """
        Delete a memory by its key.

        Args:
            key: The unique key of the memory to delete.

        Returns:
            True if successful, False otherwise.
        """
        try:
            conn = self._get_connection()
            conn.execute("DELETE FROM memory WHERE key = ?", (key,))
            conn.commit()
            logger.info(f"Deleted memory with key: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}", exc_info=True)
            return False

    def close(self):
        """Close all thread-local connections"""
        if hasattr(thread_local, "connection"):
            try:
                conn = thread_local.connection
                conn.close()
                del thread_local.connection
            except Exception as e:
                logger.error(f"Error closing connection: {e}", exc_info=True)


# --- Example Usage (for testing) ---
if __name__ == "__main__":
    print("Testing Memory Manager...")
    memory_manager = MemoryManager()

    # Add some test memories with varying importance
    memory_manager.add_memory(
        key="goal_1",
        value="User goal: Analyze stock market trends.",
        memory_type="goal",
        importance=8,
    )

    memory_manager.add_memory(
        key="plan_1",
        value="Initial plan: Gather historical stock data.",
        memory_type="plan",
        agent_name="hermes",
        importance=6,
    )

    memory_manager.add_memory(
        key="discovery_1",
        value="Found API for real-time stock quotes.",
        memory_type="discovery",
        agent_name="web_researcher",
        importance=7,
    )

    memory_manager.add_memory(
        key="error_1",
        value="Error: API key expired.",
        memory_type="error",
        agent_name="hermes",
        importance=9,
    )

    memory_manager.add_memory(
        key="solution_1",
        value="Solution: Updated API key.",
        memory_type="solution",
        agent_name="hermes",
        importance=8,
    )

    memory_manager.add_memory(
        key="idea_1",
        value="Low importance note: Consider adding sentiment analysis later.",
        memory_type="idea",
        importance=3,
    )

    # --- Test Retrievals ---
    print("\nRetrieving a specific memory:")
    memory = memory_manager.retrieve_memory("goal_1")
    if memory:
        print(f"- {memory['value']} (Importance: {memory['importance']})")

    print("\nRetrieving memories by memory_type 'error':")
    memories = memory_manager.retrieve_memories(memory_type="error")
    for mem in memories:
        print(f"- {mem['value']} (Importance: {mem['importance']})")

    print("\nRetrieving memories with minimum importance 7:")
    memories = memory_manager.retrieve_memories(min_importance=7)
    for mem in memories:
        print(f"- {mem['value']} (Importance: {mem['importance']})")

    print("\nSearching memories for 'API':")
    memories = memory_manager.search_memories("API")
    for mem in memories:
        print(f"- {mem['value']} (Importance: {mem['importance']})")

    # Clean up
    memory_manager.close()
