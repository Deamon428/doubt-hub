import os
import json
import re
import sqlite3
from typing import Any, Dict, List
from datetime import datetime
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

DB_NAME = "doubts.db"

def get_embedding(text: str) -> List[float]:
    result = genai.embed_content(model="models/text-embedding-004", content=text)
    return result["embedding"]

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

def query_gemini_tie_breaker(existing_title: str, new_doubt: str) -> bool:
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
            generation_config={"response_mime_type": "application/json"}
        )
        response = model.generate_content(prompt)
        raw_output = response.text.strip()
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
        else:
            parsed = json.loads(raw_output)
        return bool(parsed.get("is_same_roadblock", False))
    except Exception as e:
        print(f"[DoubtHub Stage 2] Gemini tie-breaker error: {e}")
        return False

# Backward compatibility alias
query_qwen_tie_breaker = query_gemini_tie_breaker

def _update_cluster_centroid(conn: sqlite3.Connection, cluster: Any, new_embedding: List[float]) -> int:
    cluster_id = cluster['id']
    current_upvotes = cluster['upvotes']
    old_centroid = np.array(json.loads(cluster['centroid_embedding']), dtype=float)
    new_vec = np.array(new_embedding, dtype=float)

    new_centroid = (old_centroid * current_upvotes + new_vec) / (current_upvotes + 1)
    norm = np.linalg.norm(new_centroid)
    if norm > 0:
        new_centroid = new_centroid / norm

    upd_cursor = conn.execute(
        'UPDATE clusters SET upvotes = upvotes + 1, centroid_embedding = ? WHERE id = ?',
        (json.dumps(new_centroid.tolist()), cluster_id)
    )
    upd_cursor.close()
    return cluster_id

def _create_cluster(conn: sqlite3.Connection, course_id: str, sheet_no: int, question_no: str, title: str, embedding: List[float]) -> int:
    ins_cursor = conn.execute(
        'INSERT INTO clusters (course_id, sheet_no, question_no, title, upvotes, centroid_embedding) VALUES (?, ?, ?, ?, ?, ?)',
        (course_id, sheet_no, question_no, title, 1, json.dumps(embedding))
    )
    cluster_id = ins_cursor.lastrowid
    ins_cursor.close()
    return cluster_id

def assign_or_create_cluster(conn: sqlite3.Connection, doubt_data: Dict[str, Any]) -> int:
    course_id = str(doubt_data['course_id']).strip().upper()
    sheet_no = int(doubt_data['sheet_no'])
    question_no = str(doubt_data['question_no']).strip().lower()
    text_to_embed = str(doubt_data['doubt_text']).strip()
    ai_failure_reason = str(doubt_data.get('ai_failure_reason', '')).strip()

    print(f"[DoubtHub] Processing: Course={course_id}, Sheet={sheet_no}, Q={question_no}")

    new_embedding = get_embedding(text_to_embed)

    cursor = conn.execute(
        'SELECT id, centroid_embedding, upvotes, title FROM clusters WHERE UPPER(TRIM(course_id)) = ? AND sheet_no = ? AND LOWER(TRIM(question_no)) = ? AND is_resolved = 0',
        (course_id, sheet_no, question_no)
    )
    existing_clusters = cursor.fetchall()
    cursor.close()

    print(f"[DoubtHub] Found {len(existing_clusters)} existing cluster(s) in DB for this question")

    if not existing_clusters:
        cluster_id = _create_cluster(conn, course_id, sheet_no, question_no, text_to_embed, new_embedding)
        print(f"[DoubtHub Stage 1] No existing clusters found. Created new Cluster #{cluster_id}")
    else:
        max_similarity = -1.0
        best_candidate = None

        for cluster in existing_clusters:
            centroid_embedding = json.loads(cluster['centroid_embedding'])
            similarity = cosine_similarity(new_embedding, centroid_embedding)
            print(f"[DoubtHub] Comparing with Cluster #{cluster['id']} -> Cosine Similarity: {similarity:.4f}")
            if similarity > max_similarity:
                max_similarity = similarity
                best_candidate = cluster

        best_id = best_candidate['id']

        if max_similarity >= 0.78:
            print(f"[DoubtHub Stage 1] High similarity ({max_similarity:.4f} >= 0.78). Auto-merging into Cluster #{best_id}")
            cluster_id = _update_cluster_centroid(conn, best_candidate, new_embedding)
        elif max_similarity < 0.50:
            print(f"[DoubtHub Stage 1] Low similarity ({max_similarity:.4f} < 0.50). Auto-creating new cluster")
            cluster_id = _create_cluster(conn, course_id, sheet_no, question_no, text_to_embed, new_embedding)
        else:
            print(f"[DoubtHub Stage 1] Gray zone similarity ({max_similarity:.4f}). Escalating to Gemini tie-breaker...")
            is_same = query_gemini_tie_breaker(best_candidate['title'], text_to_embed)
            if is_same:
                print(f"[DoubtHub Stage 2] Gemini confirmed match! Merging into Cluster #{best_id}")
                cluster_id = _update_cluster_centroid(conn, best_candidate, new_embedding)
            else:
                print(f"[DoubtHub Stage 2] Gemini rejected match. Creating new cluster")
                cluster_id = _create_cluster(conn, course_id, sheet_no, question_no, text_to_embed, new_embedding)

    d_cursor = conn.execute(
        'INSERT INTO doubts (course_id, sheet_no, question_no, doubt_text, ai_failure_reason, embedding, cluster_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        (course_id, sheet_no, question_no, text_to_embed, ai_failure_reason, json.dumps(new_embedding), cluster_id, datetime.now().isoformat())
    )
    d_cursor.close()
    conn.commit()

    return cluster_id
