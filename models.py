from pydantic import BaseModel, ConfigDict
from typing import Optional, Generator
import sqlite3
from contextlib import contextmanager
from datetime import datetime

DB_NAME = "doubts.db"

class DoubtCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    course_id: str
    sheet_no: int
    question_no: str
    doubt_text: str
    ai_failure_reason: str

class DoubtOut(BaseModel):
    id: int
    course_id: str
    sheet_no: int
    question_no: str
    doubt_text: str
    ai_failure_reason: str
    cluster_id: Optional[int]
    created_at: str

class ClusterOut(BaseModel):
    id: int
    course_id: str
    sheet_no: int
    question_no: str
    title: str
    ai_summary: Optional[str] = None
    upvotes: int
    is_resolved: int = 0
    ta_response: Optional[str] = None

class TALoginRequest(BaseModel):
    email: str
    password: str

class TAResponseInput(BaseModel):
    response_text: str
    mark_resolved: bool = False

class CommunitySolutionOut(BaseModel):
    id: int
    cluster_id: int
    author_name: str = 'Anonymous Hosteller'
    note_text: Optional[str] = None
    image_filename: Optional[str] = None
    upvotes: int = 0
    created_at: str

@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
    finally:
        conn.close()

def init_db() -> None:
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_id TEXT,
                sheet_no INTEGER,
                question_no TEXT,
                title TEXT,
                ai_summary TEXT,
                upvotes INTEGER DEFAULT 1,
                centroid_embedding TEXT,
                is_resolved INTEGER DEFAULT 0,
                ta_response TEXT
            )
        ''')
        cursor = conn.execute("PRAGMA table_info(clusters)")
        try:
            columns = [row[1] for row in cursor.fetchall()]
        finally:
            cursor.close()

        if "is_resolved" not in columns:
            alter_cursor = conn.execute("ALTER TABLE clusters ADD COLUMN is_resolved INTEGER DEFAULT 0")
            alter_cursor.close()

        if "ta_response" not in columns:
            alter_cursor = conn.execute("ALTER TABLE clusters ADD COLUMN ta_response TEXT")
            alter_cursor.close()

        conn.execute('''
            CREATE TABLE IF NOT EXISTS doubts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_id TEXT,
                sheet_no INTEGER,
                question_no TEXT,
                doubt_text TEXT,
                ai_failure_reason TEXT,
                embedding TEXT,
                cluster_id INTEGER REFERENCES clusters(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS community_solutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER REFERENCES clusters(id),
                author_name TEXT DEFAULT 'Anonymous Hosteller',
                note_text TEXT,
                image_filename TEXT,
                upvotes INTEGER DEFAULT 0,
                created_at TEXT
            )
        ''')
        conn.commit()
