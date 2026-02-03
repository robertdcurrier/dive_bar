#!/usr/bin/env python3
"""DuckDB database layer for Dive Bar."""

import hashlib
import threading
import uuid
from pathlib import Path

import duckdb


SESSIONS_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR PRIMARY KEY,
    started_at TIMESTAMP DEFAULT current_timestamp,
    ended_at TIMESTAMP,
    bar_name VARCHAR,
    agent_count INTEGER,
    config_hash VARCHAR
)
"""

MESSAGES_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    message_id VARCHAR PRIMARY KEY,
    session_id VARCHAR,
    turn_number INTEGER NOT NULL,
    agent_name VARCHAR NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp,
    model_name VARCHAR,
    tokens_prompt INTEGER,
    tokens_completion INTEGER,
    generation_time_ms REAL,
    temperature REAL,
    top_p REAL,
    selection_reason VARCHAR,
    chattiness REAL,
    score REAL,
    addressed_by VARCHAR
)
"""

AGENT_STATES_SQL = """
CREATE TABLE IF NOT EXISTS agent_states (
    session_id VARCHAR,
    agent_name VARCHAR,
    turn_number INTEGER,
    context_hash VARCHAR,
    token_count INTEGER,
    PRIMARY KEY (
        session_id, agent_name, turn_number
    )
)
"""

INSERT_MESSAGE_SQL = """
INSERT INTO messages (
    message_id, session_id, turn_number,
    agent_name, content, model_name,
    tokens_prompt, tokens_completion,
    generation_time_ms, temperature, top_p,
    selection_reason, chattiness, score,
    addressed_by
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class Database:
    """Thread-safe DuckDB wrapper for conversation logging."""

    def __init__(self, db_path: str):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path))
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist."""
        with self._lock:
            self.con.execute(SESSIONS_SQL)
            self.con.execute(MESSAGES_SQL)
            self.con.execute(AGENT_STATES_SQL)

    def start_session(
        self,
        bar_name: str,
        agent_count: int,
        config_hash: str = "",
    ) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        with self._lock:
            self.con.execute(
                "INSERT INTO sessions "
                "(session_id, bar_name, "
                "agent_count, config_hash) "
                "VALUES (?, ?, ?, ?)",
                [
                    session_id,
                    bar_name,
                    agent_count,
                    config_hash,
                ],
            )
        return session_id

    def log_message(
        self,
        session_id: str,
        turn_number: int,
        agent_name: str,
        content: str,
        model_name: str = "",
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
        generation_time_ms: float = 0.0,
        temperature: float = 0.0,
        top_p: float = 0.0,
        selection_reason: str = "",
        chattiness: float = 0.0,
        score: float = 0.0,
        addressed_by: str = "",
    ):
        """Log a single message to the database."""
        message_id = str(uuid.uuid4())
        with self._lock:
            self.con.execute(
                INSERT_MESSAGE_SQL,
                [
                    message_id,
                    session_id,
                    turn_number,
                    agent_name,
                    content,
                    model_name,
                    tokens_prompt,
                    tokens_completion,
                    generation_time_ms,
                    temperature,
                    top_p,
                    selection_reason,
                    chattiness,
                    score,
                    addressed_by,
                ],
            )

    def end_session(self, session_id: str):
        """Mark a session as ended."""
        with self._lock:
            self.con.execute(
                "UPDATE sessions "
                "SET ended_at = current_timestamp "
                "WHERE session_id = ?",
                [session_id],
            )

    def get_session_messages(
        self, session_id: str
    ) -> list[dict]:
        """Retrieve all messages for a session."""
        with self._lock:
            result = self.con.execute(
                "SELECT turn_number, agent_name, "
                "content, created_at "
                "FROM messages "
                "WHERE session_id = ? "
                "ORDER BY turn_number",
                [session_id],
            ).fetchall()
        return [
            {
                "turn_number": row[0],
                "agent_name": row[1],
                "content": row[2],
                "created_at": row[3],
            }
            for row in result
        ]

    def config_hash(self, config_str: str) -> str:
        """Generate a hash of the config for tracking."""
        return hashlib.sha256(
            config_str.encode()
        ).hexdigest()[:16]

    def close(self):
        """Close the database connection."""
        with self._lock:
            self.con.close()
