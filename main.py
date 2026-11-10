import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from models import DoubtCreate, TALoginRequest, TAResponseInput, get_db, init_db
from cluster import assign_or_create_cluster

UPLOAD_DIR = os.path.join("static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

@app.on_event("startup")
def startup_event() -> None:
    init_db()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/static/index.html")

@app.post("/api/doubts")
def create_doubt(doubt: DoubtCreate) -> Dict[str, Any]:
    with get_db() as conn:
        cluster_id = assign_or_create_cluster(conn, doubt.model_dump() if hasattr(doubt, "model_dump") else doubt.dict())
    return {"status": "success", "cluster_id": cluster_id}

@app.post("/api/clusters/{cluster_id}/upvote")
def upvote_cluster(cluster_id: int) -> Dict[str, str]:
    with get_db() as conn:
        cursor = conn.execute("UPDATE clusters SET upvotes = upvotes + 1 WHERE id = ?", (cluster_id,))
        conn.commit()
        cursor.close()
    return {"status": "success"}

@app.patch("/api/clusters/{cluster_id}/resolve")
def resolve_cluster(cluster_id: int) -> Dict[str, str]:
    with get_db() as conn:
        cursor = conn.execute("UPDATE clusters SET is_resolved = 1 WHERE id = ?", (cluster_id,))
        conn.commit()
        rowcount = cursor.rowcount
        cursor.close()
        if rowcount == 0:
            raise HTTPException(status_code=404, detail="Cluster not found")
    return {"status": "success"}

@app.post("/api/ta/login")
def ta_login(cred: TALoginRequest) -> Dict[str, str]:
    if cred.email == "test@thapar.edu" and cred.password == "test123":
        return {"status": "success", "token": "mock-ta-token-active"}
    raise HTTPException(status_code=401, detail="Invalid university credentials")

@app.post("/api/clusters/{cluster_id}/ta-response")
def post_ta_response(cluster_id: int, payload: TAResponseInput) -> Dict[str, str]:
    with get_db() as conn:
        if payload.mark_resolved:
            cursor = conn.execute(
                "UPDATE clusters SET ta_response = ?, is_resolved = 1 WHERE id = ?",
                (payload.response_text, cluster_id)
            )
        else:
            cursor = conn.execute(
                "UPDATE clusters SET ta_response = ? WHERE id = ?",
                (payload.response_text, cluster_id)
            )
        conn.commit()
        rowcount = cursor.rowcount
        cursor.close()
        if rowcount == 0:
            raise HTTPException(status_code=404, detail="Cluster not found")
    return {"status": "success"}

@app.get("/api/clusters/{cluster_id}/solutions")
def get_cluster_solutions(cluster_id: int) -> List[Dict[str, Any]]:
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT * FROM community_solutions WHERE cluster_id = ? ORDER BY upvotes DESC, id DESC",
            (cluster_id,)
        )
        solutions = [dict(row) for row in cursor.fetchall()]
        cursor.close()
    return solutions

@app.post("/api/clusters/{cluster_id}/solutions")
async def create_solution(
    cluster_id: int,
    note_text: Optional[str] = Form(None),
    author_name: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
) -> Dict[str, Any]:
    with get_db() as conn:
        chk_cur = conn.execute("SELECT id FROM clusters WHERE id = ?", (cluster_id,))
        cluster = chk_cur.fetchone()
        chk_cur.close()
        if not cluster:
            raise HTTPException(status_code=404, detail="Cluster not found")

        image_filename: Optional[str] = None
        if file and file.filename:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                ext = ".jpg"
            unique_name = f"{uuid.uuid4().hex}{ext}"
            file_dest = os.path.join(UPLOAD_DIR, unique_name)
            file_bytes = await file.read()
            with open(file_dest, "wb") as f:
                f.write(file_bytes)
            image_filename = f"/static/uploads/{unique_name}"

        clean_author = (author_name.strip() if author_name and author_name.strip() else "Anonymous Hosteller")
        clean_note = note_text.strip() if note_text and note_text.strip() else None

        if not clean_note and not image_filename:
            raise HTTPException(status_code=400, detail="Must provide either text note or an image")

        created_at = datetime.now().isoformat()
        cursor = conn.execute(
            "INSERT INTO community_solutions (cluster_id, author_name, note_text, image_filename, upvotes, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (cluster_id, clean_author, clean_note, image_filename, 0, created_at)
        )
        conn.commit()
        solution_id = cursor.lastrowid
        cursor.close()

    return {
        "id": solution_id,
        "cluster_id": cluster_id,
        "author_name": clean_author,
        "note_text": clean_note,
        "image_filename": image_filename,
        "upvotes": 0,
        "created_at": created_at
    }

@app.post("/api/solutions/{solution_id}/upvote")
def upvote_solution(solution_id: int) -> Dict[str, Any]:
    with get_db() as conn:
        cursor = conn.execute("UPDATE community_solutions SET upvotes = upvotes + 1 WHERE id = ?", (solution_id,))
        conn.commit()
        rowcount = cursor.rowcount
        cursor.close()
        if rowcount == 0:
            raise HTTPException(status_code=404, detail="Solution not found")
        cur = conn.execute("SELECT upvotes FROM community_solutions WHERE id = ?", (solution_id,))
        row = cur.fetchone()
        new_upvotes = row["upvotes"] if row else 1
        cur.close()
    return {"status": "success", "upvotes": new_upvotes}

@app.get("/api/ta/board")
def get_ta_board(course_id: str, sheet_no: int, include_resolved: bool = False) -> List[Dict[str, Any]]:
    course_id = str(course_id).strip().upper()
    sheet_no = int(sheet_no)
    with get_db() as conn:
        if include_resolved:
            cursor = conn.execute(
                "SELECT * FROM clusters WHERE UPPER(TRIM(course_id)) = ? AND sheet_no = ? ORDER BY upvotes DESC",
                (course_id, sheet_no)
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM clusters WHERE UPPER(TRIM(course_id)) = ? AND sheet_no = ? AND is_resolved = 0 ORDER BY upvotes DESC",
                (course_id, sheet_no)
            )
        clusters = [dict(row) for row in cursor.fetchall()]
        cursor.close()
        for c in clusters:
            d_cursor = conn.execute(
                "SELECT doubt_text, ai_failure_reason FROM doubts WHERE cluster_id = ?",
                (c["id"],)
            )
            c["doubts"] = [dict(r) for r in d_cursor.fetchall()]
            d_cursor.close()
    return clusters

@app.post("/api/ta/synthesize/{cluster_id}")
def synthesize_cluster(cluster_id: int) -> Dict[str, str]:
    with get_db() as conn:
        cursor = conn.execute("SELECT doubt_text, ai_failure_reason FROM doubts WHERE cluster_id = ?", (cluster_id,))
        doubts = cursor.fetchall()
        cursor.close()
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
        up_cursor = conn.execute("UPDATE clusters SET ai_summary = ? WHERE id = ?", (summary, cluster_id))
        conn.commit()
        up_cursor.close()
    return {"summary": summary}
