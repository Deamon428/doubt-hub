"""Comprehensive integration and security test suite for DoubtHub."""

import json
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import cluster
import main
import models

_BASE_VEC = [1.0, 0.0, 0.0, 0.0]
_SIMILAR_VEC = [0.98, 0.198997, 0.0, 0.0]  # cosine similarity ~ 0.98 (>= 0.78 auto-merge)
_GRAY_VEC = [0.65, 0.759934, 0.0, 0.0]     # cosine similarity ~ 0.65 (gray zone 0.50..0.78)
_ORTHOGONAL_VEC = [0.0, 0.0, 1.0, 0.0]     # cosine similarity = 0.0 (< 0.50 auto-split)


def _deterministic_embedding(text: str) -> List[float]:
    """Return controlled unit vectors based on keywords to test two-stage clustering deterministically."""
    lowered = text.lower()
    if "merge" in lowered or "same boundary" in lowered:
        return _SIMILAR_VEC
    if "gray" in lowered:
        return _GRAY_VEC
    if "different" in lowered or "orthogonal" in lowered:
        return _ORTHOGONAL_VEC
    return _BASE_VEC


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create a TestClient backed by an isolated temporary SQLite database and upload directory."""
    test_db = tmp_path / "test_doubthub.db"
    test_uploads = tmp_path / "uploads"
    test_uploads.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(models, "DB_NAME", str(test_db))
    monkeypatch.setattr(main, "UPLOAD_DIR", str(test_uploads))
    monkeypatch.setattr(cluster, "get_embedding", _deterministic_embedding)
    monkeypatch.setattr(cluster, "query_gemini_tie_breaker", lambda _a, b: "confirm" in b.lower())

    with TestClient(main.app) as test_client:
        yield test_client


def _create_sample_doubt(
    client: TestClient,
    course_id: str = "UMA023",
    sheet_no: int = 3,
    question_no: str = "4b",
    doubt_text: str = "Stuck applying boundary conditions on second-order ODE",
    ai_failure_reason: str = "AI assumed repeated roots instead of distinct roots",
) -> int:
    """Helper to post a doubt and assert cluster_id creation."""
    resp = client.post(
        "/api/doubts",
        json={
            "course_id": course_id,
            "sheet_no": sheet_no,
            "question_no": question_no,
            "doubt_text": doubt_text,
            "ai_failure_reason": ai_failure_reason,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    return int(body["cluster_id"])


# ---------------------------------------------------------------------------
# 1. Root Route & HTTP Security Headers
# ---------------------------------------------------------------------------


def test_root_returns_200_html_and_security_headers(client: TestClient) -> None:
    """GET / follows redirect to student portal HTML and carries security headers."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "DoubtHub" in resp.text

    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"


# ---------------------------------------------------------------------------
# 2. Happy-Path Creation & Filtered Retrieval
# ---------------------------------------------------------------------------


def test_create_doubt_and_retrieve_board(client: TestClient) -> None:
    """Doubts cluster cleanly and appear on case-insensitive / trimmed board queries."""
    cluster_id = _create_sample_doubt(client, course_id=" uma023 ", sheet_no=3, question_no=" 4B ")

    board_resp = client.get("/api/ta/board?course_id=uma023&sheet_no=3")
    assert board_resp.status_code == 200
    clusters = board_resp.json()
    assert len(clusters) == 1
    c = clusters[0]
    assert c["id"] == cluster_id
    assert c["course_id"] == "UMA023"
    assert c["question_no"] == "4b"
    assert c["upvotes"] == 1
    assert c["is_resolved"] == 0
    assert "centroid_embedding" not in c
    assert len(c["doubts"]) == 1

    # Upvote cluster
    upvote_resp = client.post(f"/api/clusters/{cluster_id}/upvote")
    assert upvote_resp.status_code == 200
    assert upvote_resp.json() == {"status": "success"}

    after_board = client.get("/api/ta/board?course_id=UMA023&sheet_no=3").json()
    assert after_board[0]["upvotes"] == 2


def test_peer_solution_upload_and_ta_workflow(client: TestClient) -> None:
    """Happy path for peer note/image uploads, helpful upvotes, TA login, and TA clarification."""
    cluster_id = _create_sample_doubt(client)

    # Valid PNG upload + note
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    sol_resp = client.post(
        f"/api/clusters/{cluster_id}/solutions",
        data={"author_name": "Senior Hostel-J", "note_text": "Apply Euler substitution y = e^(rx)"},
        files={"file": ("solution.png", png_bytes, "image/png")},
    )
    assert sol_resp.status_code == 200
    sol_data = sol_resp.json()
    assert sol_data["author_name"] == "Senior Hostel-J"
    assert sol_data["image_filename"].startswith("/static/uploads/")
    assert sol_data["image_filename"].endswith(".png")
    assert sol_data["manage_token"]

    # Public GET must NOT expose manage_token
    list_resp = client.get(f"/api/clusters/{cluster_id}/solutions")
    assert list_resp.status_code == 200
    sols = list_resp.json()
    assert len(sols) == 1
    assert "manage_token" not in sols[0]

    # Helpful upvote
    up_resp = client.post(f"/api/solutions/{sol_data['id']}/upvote")
    assert up_resp.status_code == 200
    assert up_resp.json()["upvotes"] == 1

    # TA login & publish official clarification
    login_resp = client.post("/api/ta/login", json={"email": "test@thapar.edu", "password": "test123"})
    assert login_resp.status_code == 200
    ta_token = login_resp.json()["token"]

    ta_note_resp = client.post(
        f"/api/clusters/{cluster_id}/ta-response",
        headers={"X-TA-Token": ta_token},
        json={"response_text": "Remember fixed-end boundary y(0)=y(L)=0", "mark_resolved": False},
    )
    assert ta_note_resp.status_code == 200

    board = client.get("/api/ta/board?course_id=UMA023&sheet_no=3").json()
    assert board[0]["ta_response"] == "Remember fixed-end boundary y(0)=y(L)=0"


def test_synthesize_cluster_with_mocked_gemini(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Whiteboard guide synthesis invokes Gemini 1.5 Flash and stores ai_summary."""
    cluster_id = _create_sample_doubt(client)

    mock_model = MagicMock()
    mock_reply = MagicMock()
    mock_reply.text = "1. Core Misconception\n2. AI trap\n3. Whiteboard steps"
    mock_model.generate_content.return_value = mock_reply
    monkeypatch.setattr(main.genai, "GenerativeModel", lambda _name: mock_model)

    resp = client.post(f"/api/ta/synthesize/{cluster_id}")
    assert resp.status_code == 200
    assert "Core Misconception" in resp.json()["summary"]


# ---------------------------------------------------------------------------
# 3. Input Validation Errors (400 / 413 / 422)
# ---------------------------------------------------------------------------


def test_validation_errors_return_standardized_error_envelope(client: TestClient) -> None:
    """Malformed inputs produce 422 JSON error envelopes rather than server errors."""
    bad_payloads: List[Dict[str, object]] = [
        {"course_id": "   ", "sheet_no": 3, "question_no": "4b", "doubt_text": "Valid doubt", "ai_failure_reason": "Valid reason"},
        {"course_id": "UMA 023;", "sheet_no": 3, "question_no": "4b", "doubt_text": "Valid doubt", "ai_failure_reason": "Valid reason"},
        {"course_id": "UMA023", "sheet_no": 0, "question_no": "4b", "doubt_text": "Valid doubt", "ai_failure_reason": "Valid reason"},
        {"course_id": "UMA023", "sheet_no": 3, "question_no": "", "doubt_text": "Valid doubt", "ai_failure_reason": "Valid reason"},
        {"course_id": "UMA023", "sheet_no": 3, "question_no": "4b", "doubt_text": "hi", "ai_failure_reason": "Valid reason"},
    ]
    for payload in bad_payloads:
        resp = client.post("/api/doubts", json=payload)
        assert resp.status_code == 422
        body = resp.json()
        assert body["status"] == "error"
        assert body["error"] == "validation_error"

    # Malformed TA login email
    bad_email_resp = client.post("/api/ta/login", json={"email": "not-an-email", "password": "test123"})
    assert bad_email_resp.status_code == 422
    assert bad_email_resp.json()["error"] == "validation_error"

    # Invalid board query parameter
    bad_query_resp = client.get("/api/ta/board?course_id=UMA023&sheet_no=0")
    assert bad_query_resp.status_code == 422


def test_upload_security_rejects_bad_types_and_oversized_files(client: TestClient) -> None:
    """Disallowed MIME types/extensions, empty payloads, and >5 MB uploads are rejected."""
    cluster_id = _create_sample_doubt(client)

    # Empty submission (no note, no file) -> 400
    empty_resp = client.post(f"/api/clusters/{cluster_id}/solutions", data={"author_name": "Peer"})
    assert empty_resp.status_code == 400

    # Disallowed HTML extension & MIME type -> 400
    html_resp = client.post(
        f"/api/clusters/{cluster_id}/solutions",
        files={"file": ("payload.html", b"<script>alert(1)</script>", "text/html")},
    )
    assert html_resp.status_code == 400

    # Spoofed MIME type on .jpg extension -> 400
    spoof_resp = client.post(
        f"/api/clusters/{cluster_id}/solutions",
        files={"file": ("payload.jpg", b"<?php echo 1; ?>", "application/x-php")},
    )
    assert spoof_resp.status_code == 400

    # Oversized > 5 MB file -> 413
    oversized_bytes = b"\x89PNG\r\n\x1a\n" + (b"A" * (5 * 1024 * 1024 + 100))
    big_resp = client.post(
        f"/api/clusters/{cluster_id}/solutions",
        files={"file": ("huge.png", oversized_bytes, "image/png")},
    )
    assert big_resp.status_code == 413


# ---------------------------------------------------------------------------
# 4. Cryptographic Token Ownership & Credential Security (401 / 403 / 200)
# ---------------------------------------------------------------------------


def test_solution_manage_token_ownership_security(client: TestClient) -> None:
    """Modifying or deleting a peer solution requires matching X-Manage-Token (403 otherwise)."""
    cluster_id = _create_sample_doubt(client)
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"

    create_resp = client.post(
        f"/api/clusters/{cluster_id}/solutions",
        data={"note_text": "Initial draft hint"},
        files={"file": ("draft.png", png_bytes, "image/png")},
    )
    assert create_resp.status_code == 200
    created = create_resp.json()
    sol_id = created["id"]
    valid_token = created["manage_token"]
    saved_name = Path(created["image_filename"]).name
    saved_file = Path(main.UPLOAD_DIR) / saved_name
    assert saved_file.exists()

    # Missing token -> 403
    assert client.patch(f"/api/solutions/{sol_id}", json={"note_text": "Hacked"}).status_code == 403
    assert client.delete(f"/api/solutions/{sol_id}").status_code == 403

    # Wrong token -> 403
    wrong_headers = {"X-Manage-Token": "wrong-token-value"}
    assert client.patch(f"/api/solutions/{sol_id}", headers=wrong_headers, json={"note_text": "Hacked"}).status_code == 403
    assert client.delete(f"/api/solutions/{sol_id}", headers=wrong_headers).status_code == 403

    # Valid token PATCH -> 200
    valid_headers = {"X-Manage-Token": valid_token}
    patch_resp = client.patch(f"/api/solutions/{sol_id}", headers=valid_headers, json={"note_text": "Corrected sign convention"})
    assert patch_resp.status_code == 200
    assert patch_resp.json()["note_text"] == "Corrected sign convention"

    # Valid token DELETE -> 200 and cleans up image on disk
    del_resp = client.delete(f"/api/solutions/{sol_id}", headers=valid_headers)
    assert del_resp.status_code == 200
    assert del_resp.json() == {"status": "success"}
    assert not saved_file.exists()
    assert client.get(f"/api/clusters/{cluster_id}/solutions").json() == []


def test_ta_auth_and_header_verification(client: TestClient) -> None:
    """Wrong TA credentials return 401; wrong X-TA-Token header on TA actions returns 403."""
    bad_login = client.post("/api/ta/login", json={"email": "test@thapar.edu", "password": "wrongpassword"})
    assert bad_login.status_code == 401

    cluster_id = _create_sample_doubt(client)

    bad_resolve = client.patch(f"/api/clusters/{cluster_id}/resolve", headers={"X-TA-Token": "bad-token"})
    assert bad_resolve.status_code == 403

    bad_ta_resp = client.post(
        f"/api/clusters/{cluster_id}/ta-response",
        headers={"X-TA-Token": "bad-token"},
        json={"response_text": "Clarification", "mark_resolved": False},
    )
    assert bad_ta_resp.status_code == 403


# ---------------------------------------------------------------------------
# 5. Domain Edge Cases: Hybrid Clustering, Centroids, Resolve, XSS, Indexes
# ---------------------------------------------------------------------------


def test_two_stage_hybrid_clustering_and_rolling_centroid(client: TestClient) -> None:
    """Verify auto-merge centroid math, gray-zone Gemini tie-breaker, and auto-split."""
    c1 = _create_sample_doubt(client, doubt_text="Initial boundary condition question")

    # High similarity (>= 0.78) auto-merges into c1 and updates upvotes + normalized centroid
    c_merge = _create_sample_doubt(client, doubt_text="Merge same boundary condition doubt")
    assert c_merge == c1

    with models.get_db() as conn:
        row = conn.execute("SELECT upvotes, centroid_embedding FROM clusters WHERE id = ?", (c1,)).fetchone()
        assert row["upvotes"] == 2
        centroid = np.array(json.loads(row["centroid_embedding"]), dtype=float)
        assert np.isclose(np.linalg.norm(centroid), 1.0, atol=1e-6)

    # Gray zone where tie-breaker rejects ("confirm" absent) -> creates new cluster
    c_gray_reject = _create_sample_doubt(client, doubt_text="Gray divergence confusion")
    assert c_gray_reject != c1

    # Orthogonal (< 0.50) auto-splits into another new cluster
    c_split = _create_sample_doubt(client, doubt_text="Completely orthogonal topic confusion")
    assert c_split not in (c1, c_gray_reject)

    # Gray zone where tie-breaker confirms match -> merges into closest candidate
    c_gray_confirm = _create_sample_doubt(client, doubt_text="Gray confirm same core issue")
    assert c_gray_confirm in (c1, c_gray_reject)


def test_resolve_toggling_and_filtering(client: TestClient) -> None:
    """Resolving a cluster hides it from active student/TA board unless include_resolved=true."""
    cid = _create_sample_doubt(client)

    assert len(client.get("/api/ta/board?course_id=UMA023&sheet_no=3").json()) == 1

    res = client.patch(f"/api/clusters/{cid}/resolve")
    assert res.status_code == 200

    assert len(client.get("/api/ta/board?course_id=UMA023&sheet_no=3").json()) == 0
    resolved_board = client.get("/api/ta/board?course_id=UMA023&sheet_no=3&include_resolved=true").json()
    assert len(resolved_board) == 1
    assert resolved_board[0]["is_resolved"] == 1


def test_xss_payload_safety_and_frontend_accessibility_guards(client: TestClient) -> None:
    """XSS strings round-trip safely and static HTML pages implement strict escaping + WCAG AA."""
    xss_payload = '<script>alert("xss")</script><img src=x onerror=alert(1)>'
    cid = _create_sample_doubt(client, doubt_text=xss_payload, ai_failure_reason=xss_payload)

    sol_resp = client.post(
        f"/api/clusters/{cid}/solutions",
        data={"author_name": xss_payload, "note_text": xss_payload},
    )
    assert sol_resp.status_code == 200

    ta_resp = client.post(
        f"/api/clusters/{cid}/ta-response",
        json={"response_text": xss_payload, "mark_resolved": False},
    )
    assert ta_resp.status_code == 200

    index_html = client.get("/static/index.html").text
    assert 'lang="en"' in index_html
    assert "escapeHTML" in index_html
    assert "loading = 'lazy'" in index_html or 'loading="lazy"' in index_html
    assert "250" in index_html
    assert 'aria-live="polite"' in index_html
    assert 'role="dialog"' in index_html
    assert "Escape" in index_html
    assert "<main>" in index_html or "<main " in index_html

    ta_html = client.get("/static/ta.html").text
    assert 'lang="en"' in ta_html
    assert "escapeHTML" in ta_html
    assert 'aria-live="polite"' in ta_html
    assert 'value="test@thapar.edu"' not in ta_html
    assert 'value="test123"' not in ta_html


def test_sqlite_indexes_and_pragmas_configured(client: TestClient) -> None:
    """Confirm composite SQLite indexes and WAL/performance pragmas exist on isolated DB."""
    client.get("/api/ta/board?course_id=UMA023&sheet_no=3")
    with models.get_db() as conn:
        indexes = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
        }
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        temp_store = conn.execute("PRAGMA temp_store;").fetchone()[0]

    assert {
        "idx_clusters_board",
        "idx_clusters_lookup",
        "idx_doubts_cluster_created",
        "idx_solutions_cluster_upvotes",
    } <= indexes
    assert str(journal_mode).lower() == "wal"
    assert int(temp_store) == 2  # MEMORY


def test_standardized_http_and_500_error_envelopes(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirm 404 and unhandled 500 errors return standardized JSON error envelopes without leaks."""
    not_found = client.patch("/api/clusters/999999/resolve")
    assert not_found.status_code == 404
    assert not_found.json() == {
        "status": "error",
        "error": "http_error",
        "detail": "Cluster not found",
    }
    assert not_found.headers.get("X-Content-Type-Options") == "nosniff"

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("sensitive internal db trace")

    monkeypatch.setattr(cluster, "get_embedding", _boom)
    with TestClient(main.app, raise_server_exceptions=False) as err_client:
        err_resp = err_client.post(
            "/api/doubts",
            json={
                "course_id": "UMA023",
                "sheet_no": 3,
                "question_no": "1a",
                "doubt_text": "Trigger unexpected error",
                "ai_failure_reason": "Testing unhandled handler",
            },
        )
    assert err_resp.status_code == 500
    body = err_resp.json()
    assert body["status"] == "error"
    assert body["error"] == "internal_server_error"
    assert "sensitive internal db trace" not in err_resp.text
