import requests
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from models import DoubtCreate, get_db, init_db
from cluster import assign_or_create_cluster

app = FastAPI()

@app.on_event("startup")
def startup_event():
    init_db()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/api/doubts")
def create_doubt(doubt: DoubtCreate):
    with get_db() as conn:
        cluster_id = assign_or_create_cluster(conn, doubt.model_dump() if hasattr(doubt, "model_dump") else doubt.dict())
    return {"status": "success", "cluster_id": cluster_id}

@app.post("/api/clusters/{cluster_id}/upvote")
def upvote_cluster(cluster_id: int):
    with get_db() as conn:
        conn.execute("UPDATE clusters SET upvotes = upvotes + 1 WHERE id = ?", (cluster_id,))
        conn.commit()
    return {"status": "success"}

@app.get("/api/ta/board")
def get_ta_board(course_id: str, sheet_no: int):
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT * FROM clusters WHERE course_id = ? AND sheet_no = ? ORDER BY upvotes DESC",
            (course_id, sheet_no)
        )
        clusters = [dict(row) for row in cursor.fetchall()]
        for c in clusters:
            d_cursor = conn.execute(
                "SELECT doubt_text, ai_failure_reason FROM doubts WHERE cluster_id = ?",
                (c["id"],)
            )
            c["doubts"] = [dict(r) for r in d_cursor.fetchall()]
    return clusters

@app.post("/api/ta/synthesize/{cluster_id}")
def synthesize_cluster(cluster_id: int):
    with get_db() as conn:
        cursor = conn.execute("SELECT doubt_text, ai_failure_reason FROM doubts WHERE cluster_id = ?", (cluster_id,))
        doubts = cursor.fetchall()
        if not doubts:
            return {"summary": "No doubts found in this cluster."}
        
        doubt_details = "\n".join([f"- Doubt: {d['doubt_text']} | Why AI failed: {d['ai_failure_reason']}" for d in doubts])
        prompt = (
            "You are an engineering teaching assistant coach. Synthesize these student doubts into a concise 3-bullet 'TA Blackboard Guide':\n"
            "1. Core Misconception\n"
            "2. Why generic AI explanations failed\n"
            "3. 2-minute whiteboard breakdown strategy\n\n"
            f"Doubts:\n{doubt_details}"
        )
        
        res = requests.post("http://127.0.0.1:11434/api/generate", json={
            "model": "qwen2.5-coder:7b",
            "prompt": prompt,
            "stream": False
        })
        summary = res.json().get("response", "")
        conn.execute("UPDATE clusters SET ai_summary = ? WHERE id = ?", (summary, cluster_id))
        conn.commit()
    return {"summary": summary}
