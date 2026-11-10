import json
import requests
import numpy as np
from typing import List, Dict

DB_NAME = "doubts.db"

def get_embedding(text: str) -> List[float]:
    response = requests.post("http://localhost:11434/api/embeddings", json={"model": "nomic-embed-text", "prompt": text})
    response.raise_for_status()
    return response.json()["embedding"]

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

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
