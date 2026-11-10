from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
from contextlib import contextmanager
import sqlite3
from datetime import datetime
import requests

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

def assign_or_create_cluster(conn, doubt_data: Dict) -> int:
    text_to_embed = f"{doubt_data['doubt_text']} | Stuck because: {doubt_data['ai_failure_reason']}"
    new_embedding = get_embedding(text_to_embed)

    conn.execute('''
        SELECT id, centroid_embedding FROM clusters WHERE course_id = ? AND sheet_no = ? AND question_no = ?
    ''', (doubt_data['course_id'], doubt_data['sheet_no'], doubt_data['question_no']))
    existing_clusters = conn.fetchall()

    max_similarity = 0.0
    cluster_id = None

    for cluster in existing_clusters:
        centroid_embedding = json.loads(cluster['centroid_embedding'])
        similarity = cosine_similarity(new_embedding, centroid_embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            cluster_id = cluster['id']

    if max_similarity >= 0.82:
        conn.execute('''
            UPDATE clusters SET upvotes = upvotes + 1 WHERE id = ?
        ''', (cluster_id,))
    else:
        title = doubt_data['doubt_text'][:60] + "..."
        conn.execute('''
            INSERT INTO clusters (course_id, sheet_no, question_no, title, upvotes, centroid_embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (doubt_data['course_id'], doubt_data['sheet_no'], doubt_data['question_no'], title, 1, json.dumps(new_embedding)))
        cluster_id = conn.lastrowid

    conn.execute('''
        INSERT INTO doubts (course_id, sheet_no, question_no, doubt_text, ai_failure_reason, embedding, cluster_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (doubt_data['course_id'], doubt_data['sheet_no'], doubt_data['question_no'], doubt_data['doubt_text'], doubt_data['ai_failure_reason'], json.dumps(new_embedding), cluster_id, datetime.now().isoformat()))
    conn.commit()

    return cluster_id

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    init_db()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/api/doubts")
async def create_doubt(doubt: DoubtCreate):
    with get_db() as conn:
        cluster_id = assign_or_create_cluster(conn, doubt.dict())
    return {"status": "success", "cluster_id": cluster_id}

@app.post("/api/clusters/{cluster_id}/upvote")
async def upvote_cluster(cluster_id: int):
    with get_db() as conn:
        conn.execute('''
            UPDATE clusters SET upvotes = upvotes + 1 WHERE id = ?
        ''', (cluster_id,))
    return {"status": "success"}

@app.get("/api/ta/board")
async def get_ta_board(course_id: str, sheet_no: int):
    with get_db() as conn:
        conn.execute('''
            SELECT * FROM clusters WHERE course_id = ? AND sheet_no = ? ORDER BY upvotes DESC
        ''', (course_id, sheet_no))
        clusters = conn.fetchall()
        cluster_list = []
        for cluster in clusters:
            cluster_dict = dict(cluster)
            cluster_dict['doubts'] = []
            conn.execute('''
                SELECT doubt_text, ai_failure_reason FROM doubts WHERE cluster_id = ?
            ''', (cluster['id'],))
            doubts = conn.fetchall()
            for doubt in doubts:
                cluster_dict['doubts'].append(dict(doubt))
            cluster_list.append(cluster_dict)
        return cluster_list

@app.post("/api/ta/synthesize/{cluster_id}")
async def synthesize_cluster(cluster_id: int):
    with get_db() as conn:
        conn.execute('''
            SELECT doubt_text FROM doubts WHERE cluster_id = ?
        ''', (cluster_id,))
        doubts = conn.fetchall()
        doubt_texts = [doubt['doubt_text'] for doubt in doubts]
        prompt = f"qwen2.5-coder:7b: Synthesize the following doubts into a 3-bullet 'TA Blackboard Guide': {doubt_texts}"
        response = requests.post("http://127.0.0.1:11434/api/generate", json={"model": "qwen2.5-coder:7b", "prompt": prompt, "stream": False})
        response.raise_for_status()
        summary = response.json()["text"]
        conn.execute('''
            UPDATE clusters SET ai_summary = ? WHERE id = ?
        ''', (summary, cluster_id))
    return {"summary": summary}
