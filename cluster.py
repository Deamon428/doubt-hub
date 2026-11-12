"""Two-stage hybrid vector and Gemini LLM clustering engine for DoubtHub."""

import json
import logging
import os
import re
import sqlite3
from datetime import datetime
from typing import Any, Dict, List

import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("doubthub.cluster")

_api_key = os.getenv("GEMINI_API_KEY")
if _api_key:
    genai.configure(api_key=_api_key)

HIGH_SIMILARITY_THRESHOLD = 0.78
LOW_SIMILARITY_THRESHOLD = 0.50


def get_embedding(text: str) -> List[float]:
    """Generate a semantic embedding vector via Gemini text-embedding-004."""
    result = genai.embed_content(model="models/text-embedding-004", content=text)
    return list(result["embedding"])


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two equal-length embedding vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    arr_a = np.asarray(vec_a, dtype=float)
    arr_b = np.asarray(vec_b, dtype=float)
    norm_a = float(np.linalg.norm(arr_a))
    norm_b = float(np.linalg.norm(arr_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(arr_a, arr_b) / (norm_a * norm_b))


def query_gemini_tie_breaker(existing_title: str, new_doubt: str) -> bool:
    """Use Gemini 1.5 Flash as a Stage-2 semantic tie-breaker for gray-zone matches."""
    prompt = (
        "You are an engineering math TA assistant.\n"
        "Determine if Doubt B describes the same underlying roadblock as Doubt A for the same problem.\n\n"
        f'Doubt A (Existing Cluster): "{existing_title}"\n'
        f'Doubt B (New Student): "{new_doubt}"\n\n'
        "Do they share the same conceptual confusion or failure mode?\n"
        'Reply ONLY with a JSON object: {"is_same_roadblock": true} or {"is_same_roadblock": false}'
    )
    try:
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json"},
        )
        response = model.generate_content(prompt)
        raw_output = response.text.strip()
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        parsed = json.loads(match.group(0) if match else raw_output)
        return bool(parsed.get("is_same_roadblock", False))
    except Exception as exc:
        logger.warning("Gemini Stage-2 tie-breaker fallback triggered: %s", exc)
        return False


# Backward compatibility alias
query_qwen_tie_breaker = query_gemini_tie_breaker


def _update_cluster_centroid(
    conn: sqlite3.Connection,
    cluster: sqlite3.Row | Dict[str, Any],
    new_embedding: List[float],
) -> int:
    """Merge a doubt into an existing cluster using a normalized rolling centroid average."""
    cluster_id = int(cluster["id"])
    current_upvotes = int(cluster["upvotes"])
    old_centroid = np.array(json.loads(cluster["centroid_embedding"]), dtype=float)
    new_vec = np.array(new_embedding, dtype=float)

    new_centroid = (old_centroid * current_upvotes + new_vec) / (current_upvotes + 1)
    norm = float(np.linalg.norm(new_centroid))
    if norm > 0.0:
        new_centroid = new_centroid / norm

    upd_cursor = conn.execute(
        "UPDATE clusters SET upvotes = upvotes + 1, centroid_embedding = ? WHERE id = ?",
        (json.dumps(new_centroid.tolist()), cluster_id),
    )
    upd_cursor.close()
    return cluster_id


def _create_cluster(
    conn: sqlite3.Connection,
    course_id: str,
    sheet_no: int,
    question_no: str,
    title: str,
    embedding: List[float],
) -> int:
    """Insert a new active roadblock cluster seeded with the initial doubt embedding."""
    ins_cursor = conn.execute(
        "INSERT INTO clusters (course_id, sheet_no, question_no, title, upvotes, centroid_embedding) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (course_id, sheet_no, question_no, title, 1, json.dumps(embedding)),
    )
    try:
        cluster_id = int(ins_cursor.lastrowid)
    finally:
        ins_cursor.close()
    return cluster_id


def assign_or_create_cluster(conn: sqlite3.Connection, doubt_data: Dict[str, Any]) -> int:
    """Assign an incoming doubt to an existing active cluster or create a new one via hybrid clustering."""
    course_id = str(doubt_data["course_id"]).strip().upper()
    sheet_no = int(doubt_data["sheet_no"])
    question_no = str(doubt_data["question_no"]).strip().lower()
    text_to_embed = str(doubt_data["doubt_text"]).strip()
    ai_failure_reason = str(doubt_data.get("ai_failure_reason", "")).strip()

    logger.info("Processing doubt: Course=%s, Sheet=%s, Q=%s", course_id, sheet_no, question_no)

    new_embedding = get_embedding(text_to_embed)

    cursor = conn.execute(
        "SELECT id, centroid_embedding, upvotes, title FROM clusters "
        "WHERE UPPER(TRIM(course_id)) = ? AND sheet_no = ? AND LOWER(TRIM(question_no)) = ? AND is_resolved = 0",
        (course_id, sheet_no, question_no),
    )
    try:
        existing_clusters = cursor.fetchall()
    finally:
        cursor.close()

    logger.info("Found %d existing active cluster(s) for question %s", len(existing_clusters), question_no)

    if not existing_clusters:
        cluster_id = _create_cluster(conn, course_id, sheet_no, question_no, text_to_embed, new_embedding)
        logger.info("Stage 1: No existing clusters found; created Cluster #%d", cluster_id)
    else:
        max_similarity = -1.0
        best_candidate = None

        for cluster in existing_clusters:
            centroid_embedding = json.loads(cluster["centroid_embedding"])
            similarity = cosine_similarity(new_embedding, centroid_embedding)
            logger.debug("Cluster #%s cosine similarity: %.4f", cluster["id"], similarity)
            if similarity > max_similarity:
                max_similarity = similarity
                best_candidate = cluster

        assert best_candidate is not None
        best_id = int(best_candidate["id"])

        if max_similarity >= HIGH_SIMILARITY_THRESHOLD:
            logger.info("Stage 1: High similarity (%.4f >= %.2f); merging into Cluster #%d", max_similarity, HIGH_SIMILARITY_THRESHOLD, best_id)
            cluster_id = _update_cluster_centroid(conn, best_candidate, new_embedding)
        elif max_similarity < LOW_SIMILARITY_THRESHOLD:
            logger.info("Stage 1: Low similarity (%.4f < %.2f); creating new cluster", max_similarity, LOW_SIMILARITY_THRESHOLD)
            cluster_id = _create_cluster(conn, course_id, sheet_no, question_no, text_to_embed, new_embedding)
        else:
            logger.info("Stage 1: Gray-zone similarity (%.4f); escalating to Gemini tie-breaker", max_similarity)
            is_same = query_gemini_tie_breaker(best_candidate["title"], text_to_embed)
            if is_same:
                logger.info("Stage 2: Gemini confirmed match; merging into Cluster #%d", best_id)
                cluster_id = _update_cluster_centroid(conn, best_candidate, new_embedding)
            else:
                logger.info("Stage 2: Gemini rejected match; creating new cluster")
                cluster_id = _create_cluster(conn, course_id, sheet_no, question_no, text_to_embed, new_embedding)

    d_cursor = conn.execute(
        "INSERT INTO doubts (course_id, sheet_no, question_no, doubt_text, ai_failure_reason, embedding, cluster_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            course_id,
            sheet_no,
            question_no,
            text_to_embed,
            ai_failure_reason,
            json.dumps(new_embedding),
            cluster_id,
            datetime.now().isoformat(),
        ),
    )
    d_cursor.close()
    conn.commit()

    return cluster_id
