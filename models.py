"""SQLite database connection, pragmas, schema migrations, and index management for DoubtHub."""

import logging
import os
import sqlite3
from contextlib import contextmanager
from typing import Generator, List

logger = logging.getLogger("doubthub.models")

DB_NAME: str = os.getenv("DB_PATH", "doubts.db")


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Yield a configured SQLite connection with WAL and performance pragmas enabled."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
    finally:
        conn.close()


_TABLE_INFO_PRAGMAS = {
    "clusters": "PRAGMA table_info(clusters)",
    "community_solutions": "PRAGMA table_info(community_solutions)",
}


def _get_table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    """Return existing column names for an allowlisted SQLite table."""
    pragma_sql = _TABLE_INFO_PRAGMAS[table_name]
    cursor = conn.execute(pragma_sql)
    try:
        return [row[1] for row in cursor.fetchall()]
    finally:
        cursor.close()


def init_db() -> None:
    """Initialize SQLite tables, backward-compatible column migrations, and composite indexes."""
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_id TEXT NOT NULL,
                sheet_no INTEGER NOT NULL,
                question_no TEXT NOT NULL,
                title TEXT NOT NULL,
                ai_summary TEXT,
                upvotes INTEGER DEFAULT 1,
                centroid_embedding TEXT,
                is_resolved INTEGER DEFAULT 0,
                ta_response TEXT
            )
            """
        )

        cluster_columns = _get_table_columns(conn, "clusters")
        if "is_resolved" not in cluster_columns:
            conn.execute("ALTER TABLE clusters ADD COLUMN is_resolved INTEGER DEFAULT 0").close()
        if "ta_response" not in cluster_columns:
            conn.execute("ALTER TABLE clusters ADD COLUMN ta_response TEXT").close()

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS doubts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_id TEXT NOT NULL,
                sheet_no INTEGER NOT NULL,
                question_no TEXT NOT NULL,
                doubt_text TEXT NOT NULL,
                ai_failure_reason TEXT NOT NULL,
                embedding TEXT,
                cluster_id INTEGER REFERENCES clusters(id) ON DELETE CASCADE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS community_solutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER REFERENCES clusters(id) ON DELETE CASCADE,
                author_name TEXT DEFAULT 'Anonymous Hosteller',
                note_text TEXT,
                image_filename TEXT,
                upvotes INTEGER DEFAULT 0,
                created_at TEXT,
                manage_token TEXT
            )
            """
        )

        solution_columns = _get_table_columns(conn, "community_solutions")
        if "manage_token" not in solution_columns:
            conn.execute("ALTER TABLE community_solutions ADD COLUMN manage_token TEXT").close()

        # Composite indexes for board queries, hybrid clustering lookup, and peer solution ranking
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_clusters_board ON clusters (course_id, sheet_no, is_resolved, upvotes DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_clusters_lookup ON clusters (UPPER(TRIM(course_id)), sheet_no, LOWER(TRIM(question_no)), is_resolved)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_doubts_cluster_created ON doubts (cluster_id, created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_solutions_cluster_upvotes ON community_solutions (cluster_id, upvotes DESC, id DESC)"
        )

        conn.commit()
        logger.debug("Database schema and indexes verified at %s", DB_NAME)
