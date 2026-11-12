"""FastAPI application entrypoint, HTTP routing, middleware, and error handling for DoubtHub."""

import hmac
import logging
import os
import re
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, AsyncIterator, Awaitable, Callable, Dict, List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from cluster import assign_or_create_cluster
from models import get_db, init_db
from schemas import (
    ClusterOut,
    CommunitySolutionCreateResponse,
    CommunitySolutionOut,
    DoubtBriefOut,
    DoubtCreate,
    DoubtCreateResponse,
    SolutionUpdateInput,
    StatusResponse,
    SynthesisResponse,
    TALoginRequest,
    TALoginResponse,
    TAResponseInput,
    UpvoteSolutionResponse,
)

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("doubthub.api")

_gemini_api_key = os.getenv("GEMINI_API_KEY")
if _gemini_api_key:
    genai.configure(api_key=_gemini_api_key)

UPLOAD_DIR: str = os.path.join("static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_BYTES: int = 5 * 1024 * 1024  # 5 MB hard limit
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
_COURSE_PARAM_PATTERN = re.compile(r"^[A-Za-z0-9_-]{2,20}$")


def _get_ta_credentials() -> tuple[str, str, str]:
    """Return configured TA email, password, and active session token."""
    return (
        os.getenv("TA_EMAIL", "test@thapar.edu"),
        os.getenv("TA_PASSWORD", "test123"),
        os.getenv("TA_TOKEN", "mock-ta-token-active"),
    )


def _constant_time_equals(provided: Optional[str], expected: Optional[str]) -> bool:
    """Compare two strings using constant-time HMAC digest verification."""
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Initialize SQLite schema and composite indexes at application startup."""
    init_db()
    logger.info("DoubtHub backend initialized.")
    yield


app = FastAPI(title="DoubtHub API", version="2.0.0", lifespan=lifespan)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.middleware("http")
async def add_security_headers(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Attach standard HTTP security headers to every response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Format HTTP exceptions into a consistent JSON error envelope."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "error": "http_error", "detail": exc.detail},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """Format Pydantic/FastAPI validation errors into a consistent JSON error envelope."""
    return JSONResponse(
        status_code=422,
        content={"status": "error", "error": "validation_error", "detail": jsonable_encoder(exc.errors())},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled server exceptions without leaking internal stack traces."""
    logger.exception("Unhandled server exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": "internal_server_error",
            "detail": "An unexpected internal error occurred.",
        },
    )


@app.get("/")
def root() -> RedirectResponse:
    """Redirect root path directly to the student escalation portal."""
    return RedirectResponse(url="/static/index.html")


@app.post("/api/doubts", response_model=DoubtCreateResponse)
def create_doubt(doubt: DoubtCreate) -> Dict[str, object]:
    """Log a student roadblock and assign or merge it into an active cluster."""
    with get_db() as conn:
        cluster_id = assign_or_create_cluster(conn, doubt.model_dump())
    return {"status": "success", "cluster_id": cluster_id}


@app.post("/api/clusters/{cluster_id}/upvote", response_model=StatusResponse)
def upvote_cluster(cluster_id: int) -> Dict[str, str]:
    """Increment the stuck-student counter for an existing cluster."""
    with get_db() as conn:
        cursor = conn.execute("UPDATE clusters SET upvotes = upvotes + 1 WHERE id = ?", (cluster_id,))
        conn.commit()
        rowcount = cursor.rowcount
        cursor.close()
        if rowcount == 0:
            raise HTTPException(status_code=404, detail="Cluster not found")
    return {"status": "success"}


@app.patch("/api/clusters/{cluster_id}/resolve", response_model=StatusResponse)
def resolve_cluster(
    cluster_id: int,
    x_ta_token: Annotated[Optional[str], Header(alias="X-TA-Token")] = None,
) -> Dict[str, str]:
    """Mark a roadblock cluster as resolved."""
    if x_ta_token is not None:
        _, _, expected_token = _get_ta_credentials()
        if not _constant_time_equals(x_ta_token, expected_token):
            raise HTTPException(status_code=403, detail="Invalid TA authorization token")

    with get_db() as conn:
        cursor = conn.execute("UPDATE clusters SET is_resolved = 1 WHERE id = ?", (cluster_id,))
        conn.commit()
        rowcount = cursor.rowcount
        cursor.close()
        if rowcount == 0:
            raise HTTPException(status_code=404, detail="Cluster not found")
    return {"status": "success"}


@app.post("/api/ta/login", response_model=TALoginResponse)
def ta_login(cred: TALoginRequest) -> Dict[str, str]:
    """Authenticate university TA credentials using constant-time digest comparison."""
    expected_email, expected_password, active_token = _get_ta_credentials()
    email_ok = _constant_time_equals(cred.email.lower(), expected_email.lower())
    password_ok = _constant_time_equals(cred.password, expected_password)
    if email_ok and password_ok:
        return {"status": "success", "token": active_token}
    raise HTTPException(status_code=401, detail="Invalid university credentials")


@app.post("/api/clusters/{cluster_id}/ta-response", response_model=StatusResponse)
def post_ta_response(
    cluster_id: int,
    payload: TAResponseInput,
    x_ta_token: Annotated[Optional[str], Header(alias="X-TA-Token")] = None,
) -> Dict[str, str]:
    """Publish an official TA clarification and optionally resolve the cluster."""
    if x_ta_token is not None:
        _, _, expected_token = _get_ta_credentials()
        if not _constant_time_equals(x_ta_token, expected_token):
            raise HTTPException(status_code=403, detail="Invalid TA authorization token")

    with get_db() as conn:
        if payload.mark_resolved:
            cursor = conn.execute(
                "UPDATE clusters SET ta_response = ?, is_resolved = 1 WHERE id = ?",
                (payload.response_text, cluster_id),
            )
        else:
            cursor = conn.execute(
                "UPDATE clusters SET ta_response = ? WHERE id = ?",
                (payload.response_text, cluster_id),
            )
        conn.commit()
        rowcount = cursor.rowcount
        cursor.close()
        if rowcount == 0:
            raise HTTPException(status_code=404, detail="Cluster not found")
    return {"status": "success"}


@app.get("/api/clusters/{cluster_id}/solutions", response_model=List[CommunitySolutionOut])
def get_cluster_solutions(cluster_id: int) -> List[Dict[str, object]]:
    """Return peer solutions for a cluster ordered by upvotes without exposing manage tokens."""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT id, cluster_id, author_name, note_text, image_filename, upvotes, created_at "
            "FROM community_solutions WHERE cluster_id = ? ORDER BY upvotes DESC, id DESC",
            (cluster_id,),
        )
        try:
            solutions = [dict(row) for row in cursor.fetchall()]
        finally:
            cursor.close()
    return solutions


@app.post("/api/clusters/{cluster_id}/solutions", response_model=CommunitySolutionCreateResponse)
async def create_solution(
    cluster_id: int,
    note_text: Annotated[Optional[str], Form()] = None,
    author_name: Annotated[Optional[str], Form()] = None,
    file: Optional[UploadFile] = File(None),
) -> Dict[str, object]:
    """Create a peer solution note and/or validated notebook photo upload."""
    with get_db() as conn:
        chk_cur = conn.execute("SELECT id FROM clusters WHERE id = ?", (cluster_id,))
        try:
            cluster = chk_cur.fetchone()
        finally:
            chk_cur.close()
        if not cluster:
            raise HTTPException(status_code=404, detail="Cluster not found")

        image_filename: Optional[str] = None
        if file and file.filename:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file extension. Allowed formats: .jpg, .jpeg, .png, .webp",
                )
            if file.content_type and file.content_type.lower() not in ALLOWED_MIME_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid MIME content type. Must be an image (jpeg, png, or webp).",
                )

            file_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(file_bytes) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Uploaded image exceeds the maximum allowed size of 5 MB.",
                )
            if len(file_bytes) == 0:
                raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

            os.makedirs(UPLOAD_DIR, exist_ok=True)
            unique_name = f"{uuid.uuid4().hex}{ext}"
            file_dest = os.path.join(UPLOAD_DIR, unique_name)
            with open(file_dest, "wb") as out_file:
                out_file.write(file_bytes)
            image_filename = f"/static/uploads/{unique_name}"

        clean_author = (
            author_name.strip()[:80]
            if author_name and author_name.strip()
            else "Anonymous Hosteller"
        )
        clean_note = note_text.strip()[:2000] if note_text and note_text.strip() else None

        if not clean_note and not image_filename:
            raise HTTPException(status_code=400, detail="Must provide either text note or an image")

        created_at = datetime.now().isoformat()
        manage_token = secrets.token_urlsafe(32)
        cursor = conn.execute(
            "INSERT INTO community_solutions "
            "(cluster_id, author_name, note_text, image_filename, upvotes, created_at, manage_token) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (cluster_id, clean_author, clean_note, image_filename, 0, created_at, manage_token),
        )
        conn.commit()
        solution_id = int(cursor.lastrowid)
        cursor.close()

    return {
        "id": solution_id,
        "cluster_id": cluster_id,
        "author_name": clean_author,
        "note_text": clean_note,
        "image_filename": image_filename,
        "upvotes": 0,
        "created_at": created_at,
        "manage_token": manage_token,
    }


def _extract_manage_token(
    x_manage_token: Optional[str],
    manage_token: Optional[str],
) -> Optional[str]:
    """Return the first non-empty manage token header value supplied by the client."""
    return x_manage_token or manage_token


@app.patch("/api/solutions/{solution_id}", response_model=CommunitySolutionOut)
def update_solution(
    solution_id: int,
    payload: SolutionUpdateInput,
    x_manage_token: Annotated[Optional[str], Header(alias="X-Manage-Token")] = None,
    manage_token: Annotated[Optional[str], Header(alias="manage_token")] = None,
) -> Dict[str, object]:
    """Update a peer solution's note text after verifying its device manage_token."""
    provided_token = _extract_manage_token(x_manage_token, manage_token)
    if not provided_token:
        raise HTTPException(status_code=403, detail="Missing X-Manage-Token ownership header")

    with get_db() as conn:
        cur = conn.execute(
            "SELECT id, cluster_id, author_name, note_text, image_filename, upvotes, created_at, manage_token "
            "FROM community_solutions WHERE id = ?",
            (solution_id,),
        )
        try:
            row = cur.fetchone()
        finally:
            cur.close()

        if not row:
            raise HTTPException(status_code=404, detail="Solution not found")
        if not _constant_time_equals(provided_token, row["manage_token"]):
            raise HTTPException(status_code=403, detail="Invalid manage token")

        upd_cur = conn.execute(
            "UPDATE community_solutions SET note_text = ? WHERE id = ?",
            (payload.note_text, solution_id),
        )
        conn.commit()
        upd_cur.close()

        result = {k: row[k] for k in ("id", "cluster_id", "author_name", "image_filename", "upvotes", "created_at")}
        result["note_text"] = payload.note_text
    return result


@app.delete("/api/solutions/{solution_id}", response_model=StatusResponse)
def delete_solution(
    solution_id: int,
    x_manage_token: Annotated[Optional[str], Header(alias="X-Manage-Token")] = None,
    manage_token: Annotated[Optional[str], Header(alias="manage_token")] = None,
) -> Dict[str, str]:
    """Delete a peer solution and any uploaded image after verifying its manage_token."""
    provided_token = _extract_manage_token(x_manage_token, manage_token)
    if not provided_token:
        raise HTTPException(status_code=403, detail="Missing X-Manage-Token ownership header")

    with get_db() as conn:
        cur = conn.execute(
            "SELECT id, image_filename, manage_token FROM community_solutions WHERE id = ?",
            (solution_id,),
        )
        try:
            row = cur.fetchone()
        finally:
            cur.close()

        if not row:
            raise HTTPException(status_code=404, detail="Solution not found")
        if not _constant_time_equals(provided_token, row["manage_token"]):
            raise HTTPException(status_code=403, detail="Invalid manage token")

        image_filename = row["image_filename"]
        del_cur = conn.execute("DELETE FROM community_solutions WHERE id = ?", (solution_id,))
        conn.commit()
        del_cur.close()

    if image_filename and isinstance(image_filename, str):
        safe_name = os.path.basename(image_filename)
        candidate_path = os.path.join(UPLOAD_DIR, safe_name)
        if os.path.isfile(candidate_path):
            os.remove(candidate_path)

    return {"status": "success"}


@app.post("/api/solutions/{solution_id}/upvote", response_model=UpvoteSolutionResponse)
def upvote_solution(solution_id: int) -> Dict[str, object]:
    """Increment the helpfulness upvote count for a community solution."""
    with get_db() as conn:
        cursor = conn.execute(
            "UPDATE community_solutions SET upvotes = upvotes + 1 WHERE id = ?",
            (solution_id,),
        )
        conn.commit()
        rowcount = cursor.rowcount
        cursor.close()
        if rowcount == 0:
            raise HTTPException(status_code=404, detail="Solution not found")
        cur = conn.execute("SELECT upvotes FROM community_solutions WHERE id = ?", (solution_id,))
        try:
            row = cur.fetchone()
            new_upvotes = int(row["upvotes"]) if row else 1
        finally:
            cur.close()
    return {"status": "success", "upvotes": new_upvotes}


@app.get("/api/ta/board", response_model=List[ClusterOut])
def get_ta_board(
    course_id: Annotated[str, Query(min_length=1, max_length=30)],
    sheet_no: Annotated[int, Query(ge=1, le=100)],
    include_resolved: bool = False,
) -> List[Dict[str, object]]:
    """Fetch roadblock clusters and batched student breakdowns for a course and tutorial sheet."""
    normalized_course = course_id.strip().upper()
    if not _COURSE_PARAM_PATTERN.match(normalized_course):
        raise HTTPException(
            status_code=422,
            detail="course_id must be 2-20 alphanumeric, hyphen, or underscore characters",
        )

    with get_db() as conn:
        if include_resolved:
            cursor = conn.execute(
                "SELECT id, course_id, sheet_no, question_no, title, ai_summary, upvotes, is_resolved, ta_response "
                "FROM clusters WHERE UPPER(TRIM(course_id)) = ? AND sheet_no = ? ORDER BY upvotes DESC",
                (normalized_course, sheet_no),
            )
        else:
            cursor = conn.execute(
                "SELECT id, course_id, sheet_no, question_no, title, ai_summary, upvotes, is_resolved, ta_response "
                "FROM clusters WHERE UPPER(TRIM(course_id)) = ? AND sheet_no = ? AND is_resolved = 0 "
                "ORDER BY upvotes DESC",
                (normalized_course, sheet_no),
            )
        try:
            clusters = [dict(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

        if not clusters:
            return []

        cluster_ids = [int(c["id"]) for c in clusters]
        doubts_by_cluster: Dict[int, List[Dict[str, str]]] = {cid: [] for cid in cluster_ids}
        chunk_size = 500
        for idx in range(0, len(cluster_ids), chunk_size):
            chunk = cluster_ids[idx : idx + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            d_cursor = conn.execute(
                f"SELECT cluster_id, doubt_text, ai_failure_reason FROM doubts WHERE cluster_id IN ({placeholders})",
                chunk,
            )
            try:
                for d_row in d_cursor.fetchall():
                    cid = int(d_row["cluster_id"])
                    doubts_by_cluster[cid].append(
                        {"doubt_text": d_row["doubt_text"], "ai_failure_reason": d_row["ai_failure_reason"]}
                    )
            finally:
                d_cursor.close()

        for cluster in clusters:
            cluster["doubts"] = doubts_by_cluster.get(int(cluster["id"]), [])

    return clusters


@app.post("/api/ta/synthesize/{cluster_id}", response_model=SynthesisResponse)
def synthesize_cluster(cluster_id: int) -> Dict[str, str]:
    """Synthesize a 3-bullet TA blackboard guide for a cluster via Gemini 1.5 Flash."""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT doubt_text, ai_failure_reason FROM doubts WHERE cluster_id = ?",
            (cluster_id,),
        )
        try:
            doubts = cursor.fetchall()
        finally:
            cursor.close()
        if not doubts:
            return {"summary": "No doubts found in this cluster."}

        doubt_details = "\n".join(
            f"- Doubt: {d['doubt_text']} | Why AI failed: {d['ai_failure_reason']}" for d in doubts
        )
        prompt = (
            "You are an engineering teaching assistant coach. Synthesize these student doubts into a concise 3-bullet 'TA Blackboard Guide':\n"
            "1. Core Misconception\n"
            "2. Why generic AI explanations failed\n"
            "3. 2-minute whiteboard breakdown strategy\n\n"
            f"Doubts:\n{doubt_details}"
        )

        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            summary = response.text.strip()
        except Exception as exc:
            logger.warning("Gemini synthesis failed for cluster #%d: %s", cluster_id, exc)
            summary = f"Failed to synthesize briefing: {exc}"

        up_cursor = conn.execute(
            "UPDATE clusters SET ai_summary = ? WHERE id = ?",
            (summary, cluster_id),
        )
        conn.commit()
        up_cursor.close()
    return {"summary": summary}
