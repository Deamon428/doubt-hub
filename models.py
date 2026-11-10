from pydantic import BaseModel
from typing import Optional
import sqlite3
from contextlib import contextmanager
from datetime import datetime

DB_NAME = "doubts.db"

class DoubtCreate(BaseModel):
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
    ai_summary: Optional[str]
    upvotes: int

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
    finally:
        conn.close()

def init_db():
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
                centroid_embedding TEXT
            )
        ''')
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
        conn.commit()
