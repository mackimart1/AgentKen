import re
import json
import threading
import time
import sqlite3
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Set, Union, Generator
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import math
from dataclasses import dataclass, field
import numpy as np
from sentence_transformers import SentenceTransformer
import zlib
from contextlib import contextmanager
import queue
import pickle
from cachetools import LRUCache

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_DB_DIR = "./agentk_memory_db"
COLLECTION_NAME = (
    "agentk_memories"  # Used for ChromaDB context, less relevant for SQLite direct
)
DEFAULT_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # A common, decent small model

# Memory decay and retention parameters (defaults, can be overridden by MemoryConfig)
DEFAULT_MEMORY_DECAY_RATE = (
    0.05  # Importance reduction per decay cycle for non-accessed memories
)
DEFAULT_MIN_IMPORTANCE_THRESHOLD = 1.0
DEFAULT_MAX_MEMORY_AGE_DAYS = 90
DEFAULT_CRITICAL_IMPORTANCE = 8.0  # Float for finer control
DEFAULT_MEMORY_REFRESH_INTERVAL_SECONDS = 3600  # 1 hour

# Performance parameters (defaults)
DEFAULT_CACHE_SIZE = 500
DEFAULT_DB_BATCH_SIZE = 100  # For potential bulk operations
DEFAULT_MAX_DB_CONNECTIONS = 5
DEFAULT_EMBEDDING_BATCH_SIZE = 32  # For batch encoding texts


@dataclass
class MemoryConfig:
    """Enhanced configuration for memory management"""

    db_dir: str = DEFAULT_DB_DIR
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    decay_rate: float = DEFAULT_MEMORY_DECAY_RATE
    min_importance: float = DEFAULT_MIN_IMPORTANCE_THRESHOLD
    max_age_days: int = DEFAULT_MAX_MEMORY_AGE_DAYS
    critical_importance: float = DEFAULT_CRITICAL_IMPORTANCE
    refresh_interval_seconds: int = DEFAULT_MEMORY_REFRESH_INTERVAL_SECONDS
    cache_size: int = DEFAULT_CACHE_SIZE
    db_batch_size: int = (
        DEFAULT_DB_BATCH_SIZE  # Currently conceptual, for future bulk ops
    )
    max_db_connections: int = DEFAULT_MAX_DB_CONNECTIONS
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE  # For batch embedding
    enable_compression: bool = True
    enable_semantic_search: bool = True
    enable_versioning: bool = True
    log_level: int = logging.INFO


@dataclass
class MemoryRecord:
    """Represents a memory item, typically retrieved from DB or cache."""

    key: str
    value: str  # Decompressed
    memory_type: str
    agent_name: Optional[str]
    task_info: Optional[str]
    importance: float
    created_at: datetime
    last_accessed: datetime
    access_count: int
    content_hash: str
    metadata: Optional[Dict[str, Any]]
    version: int
    id: Optional[int] = None  # Database row ID


class ConnectionPool:
    """Database connection pool"""

    def __init__(self, db_path: str, max_connections: int, timeout: float = 10.0):
        self.db_path = db_path
        self.max_connections = max_connections
        self.timeout = timeout
        self._pool = queue.Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self._initialize_pool()

    def _create_connection(self):
        conn = sqlite3.connect(
            self.db_path, timeout=self.timeout, check_same_thread=False
        )
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row  # Access columns by name
        return conn

    def _initialize_pool(self):
        for _ in range(self.max_connections):
            try:
                self._pool.put(self._create_connection(), block=False)
            except Exception as e:
                logger.error(f"Failed to create initial DB connection: {e}")
                # Depending on policy, might raise error or continue with fewer connections

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        try:
            conn = self._pool.get(timeout=self.timeout)
            yield conn
            # Perform a lightweight check to ensure connection is alive, e.g., conn.execute("SELECT 1")
            # If not alive, discard and create a new one. For simplicity, this is omitted here.
        except queue.Empty:
            # If pool is empty and timeout occurs, could dynamically create one if under a hard cap
            # Or raise an error indicating pool exhaustion
            logger.warning(
                "Connection pool timeout. Trying to create a temporary connection."
            )
            conn = (
                self._create_connection()
            )  # Fallback, potentially exceeding max_connections temporarily
            yield conn  # This connection will be closed by the `finally` of this context manager
        finally:
            if conn:
                # If connection was from pool, return it. If it was temporary, close it.
                # This logic needs refinement for temporary connections.
                # For now, always try to put it back.
                try:
                    self._pool.put(conn, block=False)
                except queue.Full:  # Pool is full (e.g. temporary conn was used)
                    conn.close()  # Close the temporary or excess connection

    def close_all(self):
        logger.info("Closing all database connections.")
        while not self._pool.empty():
            try:
                conn = self._pool.get(block=False)
                conn.close()
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error closing a DB connection: {e}")


class SemanticSearch:
    """Semantic search capabilities"""

    def __init__(self, model_name: str, batch_size: int):
        self.model_name = model_name
        self.batch_size = batch_size
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(
                f"Failed to load SentenceTransformer model '{model_name}': {e}. Semantic search will be disabled."
            )
            self.model = None
        self.embedding_cache = LRUCache(
            maxsize=1000
        )  # Cache for individual text embeddings
        self.cache_lock = threading.Lock()

    def get_embeddings(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        if not self.model:
            return [None] * len(texts)

        results: List[Optional[np.ndarray]] = [None] * len(texts)
        texts_to_encode_indices: List[int] = []
        texts_to_encode: List[str] = []

        with self.cache_lock:
            for i, text in enumerate(texts):
                if text in self.embedding_cache:
                    results[i] = self.embedding_cache[text]
                else:
                    texts_to_encode_indices.append(i)
                    texts_to_encode.append(text)

        if texts_to_encode:
            try:
                embeddings_generated = self.model.encode(
                    texts_to_encode, batch_size=self.batch_size, show_progress_bar=False
                )
                with self.cache_lock:
                    for i, original_idx in enumerate(texts_to_encode_indices):
                        embedding = embeddings_generated[i]
                        # Ensure embedding is np.ndarray (convert from torch.Tensor or other types if needed)
                        if hasattr(embedding, 'cpu') and hasattr(embedding, 'numpy'):
                            embedding = embedding.cpu().numpy()
                        elif not isinstance(embedding, np.ndarray):
                            embedding = np.array(embedding)
                        results[original_idx] = embedding
                        self.embedding_cache[texts_to_encode[i]] = embedding
            except Exception as e:
                logger.error(f"Error encoding texts for semantic search: {e}")
                # results for these texts will remain None

        return results

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        return self.get_embeddings([text])[0]

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        if emb1 is None or emb2 is None:
            return 0.0
        # Cosine similarity
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(sim) if not np.isnan(sim) else 0.0


@dataclass
class MemoryVersion:
    """Version tracking for memories"""

    value: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class EnhancedMemoryManager:
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        os.makedirs(self.config.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.config.db_dir, "memory_store.db")

        logger.setLevel(self.config.log_level)

        self.connection_pool = ConnectionPool(
            self.db_path, self.config.max_db_connections
        )
        if self.config.enable_semantic_search:
            self.semantic_search_handler = SemanticSearch(
                self.config.embedding_model_name, self.config.embedding_batch_size
            )
        else:
            self.semantic_search_handler = None

        self.executor = ThreadPoolExecutor(
            max_workers=os.cpu_count() or 1
        )  # For background tasks and async ops
        self.memory_cache: LRUCache = LRUCache(maxsize=self.config.cache_size)
        self.cache_lock = threading.RLock()  # Reentrant lock for cache operations

        self._initialize_db()
        self._maintenance_thread_stop_event = threading.Event()
        self._start_background_tasks()
        self._last_optimization_time = 0.0  # Initialize optimization time
        logger.info("EnhancedMemoryManager initialized.")

    def _initialize_db(self):
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    value BLOB NOT NULL,
                    memory_type TEXT DEFAULT 'general',
                    agent_name TEXT,
                    task_info TEXT,
                    importance REAL DEFAULT 5.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    content_hash TEXT NOT NULL,
                    embedding BLOB,
                    metadata TEXT,
                    version INTEGER DEFAULT 1
                )
            """
            )
            if self.config.enable_versioning:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_versions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        memory_id INTEGER NOT NULL,
                        value BLOB NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT,
                        version_number INTEGER NOT NULL,
                        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                    )
                """
                )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    target_id INTEGER NOT NULL,
                    relationship_type TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    metadata TEXT,
                    FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE,
                    UNIQUE(source_id, target_id, relationship_type)
                )
            """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed)"
            )
            conn.commit()
        logger.debug("Database initialized/verified.")

    def _compress_value(self, value: str) -> bytes:
        return (
            zlib.compress(value.encode("utf-8"))
            if self.config.enable_compression
            else value.encode("utf--8")
        )

    def _decompress_value(self, value_bytes: bytes) -> str:
        return (
            zlib.decompress(value_bytes).decode("utf-8")
            if self.config.enable_compression
            else value_bytes.decode("utf-8")
        )

    def _value_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            key=row["key"],
            value=self._decompress_value(row["value"]),
            memory_type=row["memory_type"],
            agent_name=row["agent_name"],
            task_info=row["task_info"],
            importance=row["importance"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            access_count=row["access_count"],
            content_hash=row["content_hash"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            version=row["version"],
        )

    def _update_cache_entry(self, record: MemoryRecord):
        with self.cache_lock:
            self.memory_cache[record.key] = record
        logger.debug(f"Cache updated for key: {record.key}")

    def add_memory_async(
        self,
        key: str,
        value: str,
        memory_type: str = "general",
        agent_name: Optional[str] = None,
        task_info: Optional[str] = None,
        importance: float = 5.0,
        metadata: Optional[Dict[str, Any]] = None,
        related_memories: Optional[List[Tuple[str, str, float]]] = None,
    ):
        """Asynchronously adds or updates a memory."""
        return self.executor.submit(
            self.add_memory,
            key,
            value,
            memory_type,
            agent_name,
            task_info,
            importance,
            metadata,
            related_memories,
        )

    def add_memory(
        self,
        key: str,
        value: str,
        memory_type: str = "general",
        agent_name: Optional[str] = None,
        task_info: Optional[str] = None,
        importance: float = 5.0,
        metadata: Optional[Dict[str, Any]] = None,
        related_memories: Optional[List[Tuple[str, str, float]]] = None,
    ) -> Optional[MemoryRecord]:
        content_hash = hashlib.md5(value.encode("utf-8")).hexdigest()
        compressed_value = self._compress_value(value)
        metadata_json = json.dumps(metadata) if metadata else None
        current_time = datetime.now()
        embedding_blob = None

        if self.semantic_search_handler:
            embedding = self.semantic_search_handler.get_embedding(value)
            if embedding is not None:
                embedding_blob = pickle.dumps(embedding)

        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, version, content_hash FROM memories WHERE key = ?",
                    (key,),
                )
                existing = cursor.fetchone()
                memory_id = None
                new_version_number = 1

                if existing:  # Update existing memory
                    memory_id = existing["id"]
                    current_version = existing["version"]

                    # Only create new version if content changed or explicitly versioning
                    if (
                        self.config.enable_versioning
                        and existing["content_hash"] != content_hash
                    ):
                        new_version_number = current_version + 1
                        # Move old main record to versions table
                        cursor.execute(
                            "SELECT value, metadata FROM memories WHERE id = ?",
                            (memory_id,),
                        )
                        old_main = cursor.fetchone()
                        cursor.execute(
                            """
                            INSERT INTO memory_versions (memory_id, value, metadata, version_number, timestamp)
                            VALUES (?, ?, ?, ?, (SELECT created_at FROM memories WHERE id = ?))
                        """,
                            (
                                memory_id,
                                old_main["value"],
                                old_main["metadata"],
                                current_version,
                                memory_id,
                            ),
                        )
                    else:  # No content change, or versioning disabled, just update metadata/importance
                        new_version_number = current_version
                        # if not versioning but content changed, new_version_number remains current_version (effectively overwrite)

                    cursor.execute(
                        """
                        UPDATE memories SET value=?, memory_type=?, agent_name=?, task_info=?, 
                        importance=?, last_accessed=?, access_count=access_count+1, content_hash=?, 
                        embedding=?, metadata=?, version=?
                        WHERE id=?
                    """,
                        (
                            compressed_value,
                            memory_type,
                            agent_name,
                            task_info,
                            importance,
                            current_time.isoformat(),
                            content_hash,
                            embedding_blob,
                            metadata_json,
                            new_version_number,
                            memory_id,
                        ),
                    )
                else:  # Insert new memory
                    cursor.execute(
                        """
                        INSERT INTO memories (key, value, memory_type, agent_name, task_info, importance,
                                         created_at, last_accessed, access_count, content_hash, embedding, metadata, version)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            key,
                            compressed_value,
                            memory_type,
                            agent_name,
                            task_info,
                            importance,
                            current_time.isoformat(),
                            current_time.isoformat(),
                            1,
                            content_hash,
                            embedding_blob,
                            metadata_json,
                            new_version_number,
                        ),
                    )
                    memory_id = cursor.lastrowid
                    if self.config.enable_versioning:  # Add first version explicitly
                        cursor.execute(
                            """
                            INSERT INTO memory_versions (memory_id, value, metadata, version_number, timestamp)
                            VALUES (?, ?, ?, ?, ?)
                        """,
                            (
                                memory_id,
                                compressed_value,
                                metadata_json,
                                new_version_number,
                                current_time.isoformat(),
                            ),
                        )

                if memory_id and related_memories:
                    self._update_relationships(cursor, memory_id, related_memories)

                conn.commit()

                # Retrieve the full record to return and cache
                cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
                new_row = cursor.fetchone()
                if new_row:
                    record = self._value_to_record(new_row)
                    self._update_cache_entry(record)
                    logger.info(
                        f"Memory '{key}' added/updated. Version: {record.version}. ID: {record.id}"
                    )
                    return record
                return None

        except sqlite3.Error as e:
            logger.error(
                f"SQLite error adding/updating memory '{key}': {e}", exc_info=True
            )
        except Exception as e:
            logger.error(
                f"Unexpected error adding/updating memory '{key}': {e}", exc_info=True
            )
        return None

    def _update_relationships(
        self,
        cursor: sqlite3.Cursor,
        source_id: int,
        related_memories: List[Tuple[str, str, float]],
    ):
        for rel_key, rel_type, strength in related_memories:
            cursor.execute("SELECT id FROM memories WHERE key = ?", (rel_key,))
            target_row = cursor.fetchone()
            if target_row:
                target_id = target_row["id"]
                # INSERT OR IGNORE to avoid error if relationship already exists, or update if needed
                cursor.execute(
                    """
                    INSERT INTO memory_relationships (source_id, target_id, relationship_type, strength)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(source_id, target_id, relationship_type) DO UPDATE SET strength=excluded.strength
                """,
                    (source_id, target_id, rel_type, strength),
                )
            else:
                logger.warning(
                    f"Cannot create relationship: Related key '{rel_key}' not found."
                )

    def get_memory(self, key: str) -> Optional[MemoryRecord]:
        with self.cache_lock:
            if key in self.memory_cache:
                cached_record = self.memory_cache[key]
                # Update last_accessed and access_count asynchronously for cached item
                memory_identifier = cached_record.id if cached_record.id is not None else str(key)
                self.executor.submit(
                    self._async_update_access_stats, memory_identifier, True
                )
                logger.debug(f"Memory '{key}' retrieved from cache.")
                return cached_record
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM memories WHERE key = ?", (key,))
                row = cursor.fetchone()
                if row:
                    record = self._value_to_record(row)
                    memory_identifier = record.id if record.id is not None else str(key)
                    self._async_update_access_stats(
                        memory_identifier, False
                    )  # Update DB access stats
                    self._update_cache_entry(record)  # Add to cache
                    logger.debug(f"Memory '{key}' retrieved from DB and cached.")
                    return record
        except Exception as e:
            logger.error(f"Error getting memory '{key}': {e}", exc_info=True)
        logger.debug(f"Memory '{key}' not found.")
        return None

    def _async_update_access_stats(
        self, memory_identifier: Union[int, str], is_id: bool
    ):
        """Updates access_count and last_accessed in DB."""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                field = "id" if is_id else "key"
                cursor.execute(
                    f"""
                    UPDATE memories 
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE {field} = ?
                """,
                    (datetime.now().isoformat(), memory_identifier),
                )
                conn.commit()
        except Exception as e:
            logger.error(
                f"Failed to async update access stats for {memory_identifier}: {e}"
            )

    def delete_memory(self, key: str) -> bool:
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                # ON DELETE CASCADE will handle versions and relationships
                cursor.execute("DELETE FROM memories WHERE key = ?", (key,))
                conn.commit()
                if cursor.rowcount > 0:
                    with self.cache_lock:
                        if key in self.memory_cache:
                            del self.memory_cache[key]
                    logger.info(f"Memory '{key}' deleted.")
                    return True
                logger.warning(f"Memory '{key}' not found for deletion.")
                return False
        except Exception as e:
            logger.error(f"Error deleting memory '{key}': {e}", exc_info=True)
            return False

    def clear_all_memories(self) -> bool:
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                # CASCADE will clear related tables if foreign keys are set up correctly
                cursor.execute("DELETE FROM memories")
                # If versioning is on, also clear memory_versions explicitly if not cascaded as expected
                if self.config.enable_versioning:
                    cursor.execute("DELETE FROM memory_versions")
                cursor.execute("DELETE FROM memory_relationships")
                conn.commit()
            with self.cache_lock:
                self.memory_cache.clear()
            logger.info("All memories cleared.")
            return True
        except Exception as e:
            logger.error(f"Error clearing all memories: {e}", exc_info=True)
            return False

    def get_all_keys(self) -> List[str]:
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT key FROM memories ORDER BY last_accessed DESC")
                return [row["key"] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting all keys: {e}", exc_info=True)
            return []

    def semantic_search(
        self, query_text: str, limit: int = 5, min_similarity: float = 0.5
    ) -> List[MemoryRecord]:
        if not self.semantic_search_handler or not self.semantic_search_handler.model:
            logger.warning("Semantic search is disabled or model not loaded.")
            return []

        query_embedding = self.semantic_search_handler.get_embedding(query_text)
        if query_embedding is None:
            return []

        candidates = []
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                # Fetch memories that have embeddings.
                # This could be slow if many memories. Consider filtering or sampling.
                cursor.execute("SELECT * FROM memories WHERE embedding IS NOT NULL")
                rows = cursor.fetchall()

            for row in rows:
                mem_embedding_blob = row["embedding"]
                if mem_embedding_blob:
                    mem_embedding = pickle.loads(mem_embedding_blob)
                    similarity = self.semantic_search_handler.compute_similarity(
                        query_embedding, mem_embedding
                    )
                    if similarity >= min_similarity:
                        record = self._value_to_record(row)
                        candidates.append((record, similarity))

            # Sort by similarity
            candidates.sort(key=lambda x: x[1], reverse=True)

            # Update access stats for retrieved memories
            for record, _ in candidates[:limit]:
                self._async_update_access_stats(record.id, True)
                self._update_cache_entry(record)  # Ensure cache is up-to-date

            return [record for record, sim in candidates[:limit]]
        except Exception as e:
            logger.error(f"Semantic search failed: {e}", exc_info=True)
            return []

    def get_related_memories(
        self,
        key: str,
        relationship_type: Optional[str] = None,
        min_strength: float = 0.1,
        limit: int = 10,
    ) -> List[MemoryRecord]:
        results = []
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM memories WHERE key = ?", (key,))
                source_row = cursor.fetchone()
                if not source_row:
                    logger.warning(
                        f"Source key '{key}' not found for getting related memories."
                    )
                    return []
                source_id = source_row["id"]

                query = """
                    SELECT m.* 
                    FROM memories m
                    JOIN memory_relationships mr ON m.id = mr.target_id
                    WHERE mr.source_id = ? AND mr.strength >= ?
                """
                params: List[Any] = [source_id, min_strength]
                if relationship_type:
                    query += " AND mr.relationship_type = ?"
                    params.append(relationship_type)
                query += " ORDER BY mr.strength DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                for row in cursor.fetchall():
                    record = self._value_to_record(row)
                    results.append(record)
                    memory_identifier = record.id if record.id is not None else str(record.key)
                    self._async_update_access_stats(memory_identifier, True)
                    self._update_cache_entry(record)

        except Exception as e:
            logger.error(
                f"Failed to get related memories for '{key}': {e}", exc_info=True
            )
        return results

    def get_memory_history(self, key: str) -> List[MemoryVersion]:
        if not self.config.enable_versioning:
            logger.info("Versioning is disabled. Cannot retrieve history.")
            return []
        history = []
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM memories WHERE key = ?", (key,))
                mem_row = cursor.fetchone()
                if not mem_row:
                    return []
                memory_id = mem_row["id"]

                cursor.execute(
                    """
                    SELECT value, timestamp, metadata FROM memory_versions 
                    WHERE memory_id = ? ORDER BY version_number DESC
                """,
                    (memory_id,),
                )

                for row in cursor.fetchall():
                    history.append(
                        MemoryVersion(
                            value=self._decompress_value(row["value"]),
                            timestamp=datetime.fromisoformat(row["timestamp"]),
                            metadata=(
                                json.loads(row["metadata"]) if row["metadata"] else None
                            ),
                        )
                    )
        except Exception as e:
            logger.error(
                f"Failed to get memory history for '{key}': {e}", exc_info=True
            )
        return history

    def _update_memory_decay(self):
        logger.info("Running memory decay update...")
        decay_threshold_date = datetime.now() - timedelta(
            days=1
        )  # Only decay if not accessed recently
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                # Decay importance for memories not recently accessed and above min importance
                # Exclude critical memories from automatic decay below critical threshold
                # Note: Importance is REAL, so direct comparison is fine
                cursor.execute(
                    f"""
                    UPDATE memories
                    SET importance = MAX(?, importance - ?) 
                    WHERE last_accessed < ? AND importance > ? AND importance < ?
                """,
                    (
                        self.config.min_importance,
                        self.config.decay_rate,
                        decay_threshold_date.isoformat(),
                        self.config.min_importance,
                        self.config.critical_importance,
                    ),
                )
                conn.commit()
                logger.info(
                    f"Memory decay applied. {cursor.rowcount} memories affected."
                )
        except Exception as e:
            logger.error(f"Error during memory decay: {e}", exc_info=True)

    def _cleanup_old_memories(self, max_age_days_override: Optional[int] = None):
        age_days = (
            max_age_days_override
            if max_age_days_override is not None
            else self.config.max_age_days
        )
        cutoff_date = datetime.now() - timedelta(days=age_days)
        logger.info(
            f"Running cleanup for memories older than {age_days} days (before {cutoff_date.isoformat()}) and below critical importance."
        )

        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                # ON DELETE CASCADE should handle versions and relationships
                cursor.execute(
                    """
                    DELETE FROM memories
                    WHERE created_at < ? AND importance < ?
                """,
                    (cutoff_date.isoformat(), self.config.critical_importance),
                )
                conn.commit()
                logger.info(
                    f"Memory cleanup completed. {cursor.rowcount} memories deleted."
                )
                # Cache will naturally expire items or they'll be overwritten.
                # For a more aggressive cache sync after mass delete, one might iterate and remove.
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}", exc_info=True)

    def get_memory_stats(self) -> Dict[str, Any]:
        stats = {
            "total_memories": 0,
            "total_versions": 0,
            "total_relationships": 0,
            "cache_size": 0,
        }
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM memories")
                stats["total_memories"] = cursor.fetchone()[0]
                if self.config.enable_versioning:
                    cursor.execute("SELECT COUNT(*) FROM memory_versions")
                    stats["total_versions"] = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM memory_relationships")
                stats["total_relationships"] = cursor.fetchone()[0]
            with self.cache_lock:
                stats["cache_size"] = len(self.memory_cache)
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}", exc_info=True)
        return stats

    def _optimize_database(self):
        logger.info("Optimizing database...")
        try:
            with self.connection_pool.get_connection() as conn:
                conn.execute("VACUUM;")
                conn.execute("ANALYZE;")  # Analyze all tables
                conn.commit()
            logger.info("Database optimization completed.")
        except Exception as e:
            logger.error(f"Database optimization failed: {e}", exc_info=True)

    def _start_background_tasks(self):
        def maintenance_loop():
            logger.info("Memory maintenance thread started.")
            # Initial delay before first run
            time.sleep(min(60, self.config.refresh_interval_seconds))

            while not self._maintenance_thread_stop_event.is_set():
                try:
                    self.executor.submit(self._update_memory_decay)
                    self.executor.submit(self._cleanup_old_memories)
                    # Optimize less frequently
                    if hasattr(self, "_last_optimization_time"):
                        if (
                            time.time() - self._last_optimization_time
                            > self.config.refresh_interval_seconds * 24
                        ):  # ~Once a day
                            self.executor.submit(self._optimize_database)
                            self._last_optimization_time = time.time()
                    else:
                        self._last_optimization_time = time.time()  # Initialize
                        self.executor.submit(self._optimize_database)

                except Exception as e:  # Catch errors from submitting tasks
                    logger.error(
                        f"Error in maintenance loop submission: {e}", exc_info=True
                    )

                # Wait for the next interval or until stop event is set
                self._maintenance_thread_stop_event.wait(
                    self.config.refresh_interval_seconds
                )
            logger.info("Memory maintenance thread stopped.")

        self._maintenance_thread = threading.Thread(
            target=maintenance_loop, daemon=True, name="MemoryMaintenanceThread"
        )
        self._maintenance_thread.start()

    def close(self):
        logger.info("Shutting down EnhancedMemoryManager...")
        self._maintenance_thread_stop_event.set()
        if hasattr(self, "_maintenance_thread") and self._maintenance_thread.is_alive():
            self._maintenance_thread.join(timeout=10)  # Wait for thread to finish
        self.executor.shutdown(wait=True)
        self.connection_pool.close_all()
        logger.info("EnhancedMemoryManager shutdown complete.")


# --- Global instance for the wrapper function ---
_global_memory_manager_instance: Optional[EnhancedMemoryManager] = None
_global_memory_manager_lock = threading.Lock()


def get_global_memory_manager(
    config: Optional[MemoryConfig] = None,
) -> EnhancedMemoryManager:
    global _global_memory_manager_instance
    if _global_memory_manager_instance is None:
        with _global_memory_manager_lock:
            if _global_memory_manager_instance is None:
                logger.info("Initializing global EnhancedMemoryManager instance.")
                _global_memory_manager_instance = EnhancedMemoryManager(config=config)
    return _global_memory_manager_instance


def memory_manager_tool(task: str) -> Dict[str, Any]:
    """
    Tool interface for interacting with the EnhancedMemoryManager.
    Parses natural language-like commands.
    """
    manager = get_global_memory_manager()  # Uses global instance
    task_lower = task.lower().strip()

    # Write: write key 'my_key' value 'my_value' [type 'type'] [importance N] [meta '{"k":"v"}']
    write_match = re.match(
        r"write key '(.+?)' value '(.+?)'(?:\s+type '(.+?)')?(?:\s+importance (\d+\.?\d*))?(?:\s+meta '(.+?)')?",
        task_lower,
    )
    if write_match:
        groups = write_match.groups()
        key, value = groups[0], groups[1]
        mem_type = groups[2] if groups[2] else "general"
        importance = float(groups[3]) if groups[3] else 5.0
        metadata = json.loads(groups[4]) if groups[4] else None

        # Using the async submission method of the manager for non-blocking behavior for this tool
        future = manager.add_memory_async(
            key, value, memory_type=mem_type, importance=importance, metadata=metadata
        )
        # For this tool, we might want to wait a short period or just confirm submission
        try:
            record = future.result(
                timeout=5.0
            )  # Wait up to 5s for completion for immediate feedback
            if record:
                return {
                    "status": "success",
                    "message": f"Memory '{key}' stored.",
                    "key": key,
                    "id": record.id,
                    "version": record.version,
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to store memory '{key}'.",
                    "key": key,
                }
        except TimeoutError:
            return {
                "status": "success",
                "message": "Memory storage submitted (processing in background).",
                "key": key,
            }
        except Exception as e:
            logger.error(f"Error processing write for key '{key}': {e}")
            return {
                "status": "error",
                "message": f"Error storing memory '{key}': {str(e)}",
                "key": key,
            }

    # Read: read key 'my_key'
    read_match = re.match(r"read key '(.+?)'", task_lower)
    if read_match:
        key = read_match.group(1)
        record = manager.get_memory(key)
        if record:
            return {"status": "success", "key": key, "record": record.__dict__}
        return {"status": "error", "message": f"Key '{key}' not found."}

    # Delete: delete key 'my_key'
    delete_match = re.match(r"delete key '(.+?)'", task_lower)
    if delete_match:
        key = delete_match.group(1)
        if manager.delete_memory(key):
            return {"status": "success", "message": f"Key '{key}' deleted."}
        return {
            "status": "error",
            "message": f"Failed to delete key '{key}' (possibly not found).",
        }

    # List keys
    if task_lower == "list keys":
        keys = manager.get_all_keys()
        return {"status": "success", "keys": keys}

    # Query: query 'search text' [limit N] [min_similarity F]
    query_match = re.match(
        r"query '(.+?)'(?:\s+limit (\d+))?(?:\s+min_similarity (\d\.\d+))?", task_lower
    )
    if query_match:
        query_text = query_match.group(1)
        limit = int(query_match.group(2)) if query_match.group(2) else 5
        min_similarity = float(query_match.group(3)) if query_match.group(3) else 0.5
        memories = manager.semantic_search(
            query_text, limit=limit, min_similarity=min_similarity
        )
        return {"status": "success", "memories": [mem.__dict__ for mem in memories]}

    # Related: related to 'source_key' [type 'rel_type'] [min_strength F] [limit N]
    related_match = re.match(
        r"related to '(.+?)'(?:\s+type '(.+?)')?(?:\s+min_strength (\d\.\d+))?(?:\s+limit (\d+))?",
        task_lower,
    )
    if related_match:
        key = related_match.group(1)
        rel_type = related_match.group(2)
        min_strength = float(related_match.group(3)) if related_match.group(3) else 0.1
        limit = int(related_match.group(4)) if related_match.group(4) else 10
        related = manager.get_related_memories(
            key, relationship_type=rel_type, min_strength=min_strength, limit=limit
        )
        return {
            "status": "success",
            "related_memories": [mem.__dict__ for mem in related],
        }

    # History: history of 'my_key'
    history_match = re.match(r"history of '(.+?)'", task_lower)
    if history_match:
        key = history_match.group(1)
        history = manager.get_memory_history(key)
        return {
            "status": "success",
            "key": key,
            "history": [v.__dict__ for v in history],
        }

    # Stats
    if task_lower == "stats":
        return {"status": "success", "stats": manager.get_memory_stats()}

    # Cleanup: cleanup [N days]
    cleanup_match = re.match(r"cleanup(?: (\d+))?(?: days)?", task_lower)
    if cleanup_match:
        days_str = cleanup_match.group(1)
        days_override = int(days_str) if days_str else None
        manager.executor.submit(
            manager._cleanup_old_memories, days_override
        )  # Run in background
        return {
            "status": "success",
            "message": f"Cleanup task submitted."
            + (
                f" (for memories older than {days_override} days)"
                if days_override
                else ""
            ),
        }

    # Clear all: clear all memories --confirm
    if task_lower == "clear all memories --confirm":
        if manager.clear_all_memories():
            return {"status": "success", "message": "All memories cleared."}
        return {"status": "error", "message": "Failed to clear all memories."}
    elif task_lower == "clear all memories":
        return {
            "status": "info",
            "message": "Confirmation needed. Use 'clear all memories --confirm'.",
        }

    return {"status": "error", "message": "Invalid task format or unknown command."}


if __name__ == "__main__":
    # Example usage and testing
    # Custom config for testing
    test_config = MemoryConfig(
        db_dir="./test_agentk_memory_db",
        embedding_model_name="all-MiniLM-L6-v2",  # Ensure this model is available or use a default one
        refresh_interval_seconds=10,  # Frequent for testing
        log_level=logging.DEBUG,
    )

    # Ensure clean start for test DB
    if os.path.exists(test_config.db_dir):
        import shutil

        shutil.rmtree(test_config.db_dir)

    # Initialize manager for direct use (or it will be created by tool)
    manager = get_global_memory_manager(config=test_config)

    logger.info("--- Testing EnhancedMemoryManager & memory_manager_tool ---")

    # Test write
    print(
        memory_manager_tool(
            "write key 'goal1' value 'Achieve world peace' type 'ambition' importance 9.5 meta '{\"source\":\"test\"}'"
        )
    )
    print(
        memory_manager_tool(
            "write key 'task1' value 'Develop a universal translator' type 'project' importance 8"
        )
    )
    print(
        memory_manager_tool(
            "write key 'fact1' value 'The sky is blue on a clear day.' type 'observation' importance 3"
        )
    )

    # Wait for async writes to settle if necessary, though add_memory_async waits for a bit
    time.sleep(1)

    # Test read
    print(memory_manager_tool("read key 'goal1'"))
    print(memory_manager_tool("read key 'nonexistent'"))

    # Test update (will create new version)
    print(
        memory_manager_tool(
            "write key 'goal1' value 'Achieve lasting world peace and prosperity' importance 9.8"
        )
    )
    time.sleep(0.1)

    # Test history
    print(memory_manager_tool("history of 'goal1'"))

    # Test list keys
    print(memory_manager_tool("list keys"))

    # Test semantic query
    print(memory_manager_tool("query 'global harmony discussions' limit 2"))

    # Test adding relationships (assuming add_memory in EMM is synchronous or we wait for future)
    # For this test, let's make add_memory synchronous in EMM.
    # If using async via tool, need to ensure 'goal1' and 'task1' are present.
    # For simplicity of test, assume they are or use manager.add_memory directly.
    manager.add_memory(
        "task1.1",
        "Research linguistic patterns",
        related_memories=[("task1", "subtask_of", 0.9)],
    )
    time.sleep(0.1)
    print(memory_manager_tool("related to 'task1'"))

    # Test stats
    print(memory_manager_tool("stats"))

    # Test delete
    print(memory_manager_tool("delete key 'fact1'"))
    print(memory_manager_tool("read key 'fact1'"))  # Should be not found

    # Test cleanup (will run in background)
    # To test cleanup effectively, we'd need to manipulate created_at or wait long time.
    # For now, just submit it.
    print(
        memory_manager_tool("cleanup 0 days")
    )  # Cleanup things older than 0 days (i.e. almost everything not critical)
    logger.info("Waiting for cleanup and decay tasks to potentially run...")
    time.sleep(
        test_config.refresh_interval_seconds + 5
    )  # Wait for one maintenance cycle + buffer

    print(memory_manager_tool("stats"))  # See if stats changed

    # Test clear all
    print(memory_manager_tool("clear all memories"))  # Needs confirmation
    print(memory_manager_tool("clear all memories --confirm"))
    print(memory_manager_tool("stats"))  # Should be empty or close to it

    logger.info("--- Test Finished ---")

    # Explicitly close the manager to stop background threads
    manager.close()

    # Clean up test DB directory
    # import shutil
    # if os.path.exists(test_config.db_dir):
    #     shutil.rmtree(test_config.db_dir)
    # logger.info(f"Cleaned up test database directory: {test_config.db_dir}")
