# DoubtHub — Collaborative Tutorial-Sheet Roadblock Engine

DoubtHub is a semantic doubt-clustering and peer-resolution platform built for Thapar University engineering tutorial sessions and hostel study workflows. Instead of losing repeated tutorial questions across scattered WhatsApp chats or watching generic AI chatbots stall on course-specific constraints, DoubtHub captures **what the student tried**, **why generic AI solutions failed (AI divergence)**, and **clusters paraphrased formulations of the exact same roadblock** into a shared TA & peer board.

---

## Campus Problem & Core Capabilities

In large engineering courses, dozens of students get stuck on the exact same step of a tutorial sheet (for example, a sign convention in a circuit problem, a boundary condition in Differential Equations, or a proof step in Discrete Math), yet each student phrases it differently or asks an LLM that hallucinates outside the instructor's rubric.

DoubtHub solves this end-to-end:

1. **AI-Failure & Roadblock Capture (`POST /api/doubts`)**: Students submit their Course ID, Tutorial Sheet #, Question #, doubt description, and optional `ai_failure_reason` explaining where an AI assistant broke down.
2. **Two-Stage Hybrid Clustering (`cluster.py`)**:
   - **Stage 1 (Semantic Vector Scan)**: Embeds incoming doubts using Gemini `models/text-embedding-004` and compares cosine similarity against active (`is_resolved = 0`) cluster centroids for the normalized `(course_id, sheet_no, question_no)` tuple:
     - `similarity >= 0.78`: Auto-merges into the matching cluster and updates the rolling centroid vector.
     - `similarity < 0.50`: Auto-creates a new distinct roadblock cluster.
   - **Stage 2 (LLM Gray-Zone Tie-Breaker)**: Similarities in the gray zone (`0.50` – `0.78`) invoke `gemini-1.5-flash` with strict JSON adjudication (`{"is_same_roadblock": true/false}`) to prevent over-splitting or false merges on math/code paraphrases.
3. **Peer Solution & Notebook Photo Sharing**: Students already on the same sheet can browse the collapsible **Active Roadblocks** drawer, upvote (`+1 I'm stuck here too`), or upload written hints and handwritten notebook photos (`static/uploads/`).
4. **TA Control Panel & AI Blackboard Synthesis (`static/ta.html`)**: TAs sign in, review live clusters ranked by student demand, generate a 3-bullet TA Blackboard Guide via `gemini-1.5-flash` (`POST /api/ta/synthesize/{cluster_id}`), post official verified responses, and mark roadblocks resolved.

---

## Architecture & Engineering Pillars

### Code Quality & Separation of Concerns
- **`schemas.py`**: Strict Pydantic v2 `BaseModel` contracts with `Field` length/range constraints and `@field_validator` sanitization for every request and response payload.
- **`models.py`**: Isolated SQLite persistence layer managing schema initialization, safe column migrations, PRAGMA tuning, and composite indexing.
- **`cluster.py`**: Gemini embedding and two-stage hybrid clustering pipeline using structured Python `logging` (`doubthub.cluster`) and explicit resource cleanup.
- **`main.py`**: FastAPI HTTP layer using `lifespan` startup initialization, structured `logging`, and centralized exception handlers returning a consistent JSON error envelope:
  ```json
  {
    "status": "error",
    "error": "http_error | validation_error | internal_server_error",
    "detail": "..."
  }
  ```

### Zero-Friction Student Access & `localStorage` Device Tokens
Forcing university students to create accounts, verify emails, or go through SSO every time they check a tutorial question during a 50-minute lab creates enough friction that students abandon the tool and fall back to unstructured messaging apps. DoubtHub intentionally adopts a **zero-friction student model**:

- **Upvote Deduplication**: Student browsers track upvoted cluster IDs (`doubthub_upvoted_clusters`) and solution IDs (`doubthub_upvoted_solutions`) in `localStorage` to prevent accidental double-voting without requiring login.
- **Creator-Scoped Peer Solution Ownership**: When a student posts a peer note or notebook photo via `POST /api/clusters/{cluster_id}/solutions`, the server generates a cryptographic `manage_token` stored in SQLite and returned once in the creation response. The student browser persists `{solutionId: manage_token}` in `localStorage` (`doubthub_solution_tokens`) and sends it via `X-Manage-Token` (or `manage_token`) when editing (`PATCH`) or deleting (`DELETE`) their own submission.
- **Privacy Guarantee**: `GET /api/clusters/{cluster_id}/solutions` strictly excludes `manage_token`, and mutation endpoints verify tokens server-side using constant-time `hmac.compare_digest()` comparison (returning `403` on missing or mismatched tokens).

### Security Hardening
- **XSS Prevention (`static/index.html`, `static/ta.html`)**: Eliminated unsafe HTML string interpolation of user-supplied text. All dynamic rendering uses `escapeHTML`, safe `textContent` DOM nodes, and regex path validation ensuring uploaded image URLs stay strictly within `/static/uploads/...`.
- **SQL Injection Defense**: 100% parameterized SQLite queries (`?` placeholders) and an allowlisted mapping (`_TABLE_INFO_PRAGMAS`) for table introspection.
- **Upload Hardening**: Enforces a 5 MB file size limit (`MAX_UPLOAD_BYTES`), extension allowlist (`.jpg`, `.jpeg`, `.png`, `.webp`), and MIME type validation (`image/jpeg`, `image/jpg`, `image/png`, `image/webp`), returning `413` or `400` on invalid uploads.
- **Constant-Time Auth**: TA credentials (`POST /api/ta/login`, defaulting to `test@thapar.edu` / `test123` unless overridden by `TA_EMAIL` / `TA_PASSWORD`) and `X-TA-Token` / `X-Manage-Token` checks use `hmac.compare_digest()`.
- **HTTP Security Headers Middleware**: Attaches `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and `Referrer-Policy: strict-origin-when-cross-origin` to every response.

### Database Efficiency & Frontend Performance
- **SQLite Pragmas**: Configures `PRAGMA journal_mode=WAL;`, `PRAGMA synchronous=NORMAL;`, `PRAGMA temp_store=MEMORY;`, and `PRAGMA foreign_keys=ON;`.
- **Composite Indexes**: Creates `idx_clusters_board`, `idx_clusters_lookup`, `idx_doubts_cluster_created`, and `idx_solutions_cluster_upvotes`.
- **N+1 Query Elimination**: `GET /api/ta/board` fetches matching clusters once (omitting heavy `centroid_embedding` blobs) and loads associated doubts in 500-item chunked `IN` queries instead of issuing one query per cluster.
- **UI Optimization**: 250 ms debouncing on Course ID / Sheet # board filters and `loading="lazy"` on peer solution images.

### Accessibility (WCAG 2.1 AA)
- Semantic landmarks (`<header>`, `<nav>`, `<main>`, `<section>`) and `<html lang="en">` across both UIs.
- Explicit `<label for="...">` bindings for all form controls, descriptive `aria-label` attributes, and meaningful image `alt` text.
- `aria-live="polite"` announcements on dynamic feeds and toast banners (`#toast-banner`, `#ta-toast`).
- Full keyboard navigation: high-contrast `:focus-visible` outlines, accessible image lightbox (`#image-modal`) with focus trap, `Escape` key dismissal, and focus restoration to the triggering control.

---

## API Reference

| Method | Endpoint | Purpose | Auth / Notes |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Redirects to `/static/index.html` | Public |
| `POST` | `/api/doubts` | Submit a new doubt and assign/create cluster | JSON (`DoubtCreate`) |
| `GET` | `/api/ta/board` | List clusters with nested doubts (`course_id`, `sheet_no`, `include_resolved`) | Public; excludes `centroid_embedding` |
| `POST` | `/api/clusters/{cluster_id}/upvote` | Increment cluster stuck count | Public |
| `PATCH` | `/api/clusters/{cluster_id}/resolve` | Mark cluster resolved (`is_resolved = 1`) | Optional `X-TA-Token` (`403` if invalid) |
| `POST` | `/api/ta/login` | Authenticate TA credentials and return session token | JSON (`TALoginRequest`) |
| `POST` | `/api/clusters/{cluster_id}/ta-response` | Post official TA response (optionally resolve) | Optional `X-TA-Token` (`403` if invalid) |
| `POST` | `/api/ta/synthesize/{cluster_id}` | Generate 3-bullet TA Blackboard Guide via `gemini-1.5-flash` | Persists `ai_summary` |
| `GET` | `/api/clusters/{cluster_id}/solutions` | List peer solutions ordered by `upvotes DESC, id DESC` | Omits `manage_token` |
| `POST` | `/api/clusters/{cluster_id}/solutions` | Upload peer note and/or notebook image (`multipart/form-data`) | Returns one-time `manage_token`; 5 MB max |
| `PATCH` | `/api/solutions/{solution_id}` | Edit a peer solution note | Requires `X-Manage-Token` (or `manage_token`) |
| `DELETE` | `/api/solutions/{solution_id}` | Delete a peer solution and its uploaded image | Requires `X-Manage-Token` (or `manage_token`) |
| `POST` | `/api/solutions/{solution_id}/upvote` | Increment peer solution upvote count | Public |

---

## Local Setup & Development

1. **Create and activate a virtual environment, then install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure environment variables (`.env`):**
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key
   # Optional overrides for TA authentication:
   # TA_EMAIL=ta@thapar.edu
   # TA_PASSWORD=your_secure_password
   # TA_SESSION_TOKEN=your_secret_ta_token
   ```

3. **Run the server locally:**
   ```bash
   uvicorn main:app --reload
   ```
   Open [http://127.0.0.1:8000](http://127.0.0.1:8000) for the Student Portal or [http://127.0.0.1:8000/static/ta.html](http://127.0.0.1:8000/static/ta.html) for the TA Control Panel.

---

## Running Automated Tests

The test suite uses FastAPI `TestClient`, isolated `tmp_path` SQLite databases/upload directories (leaving production `doubts.db` untouched), and mocked Gemini embedding/LLM calls so tests run offline deterministically:

```bash
./venv/bin/pytest -v
```

Coverage includes:
- Security headers and root redirect HTML delivery
- Doubt creation, input normalization, N+1-free board queries, and resolve filtering
- Two-stage hybrid clustering (`>= 0.78` auto-merge, `< 0.50` auto-split, gray-zone LLM tie-breaking) and rolling centroid math
- Peer solution uploads, MIME/extension/5 MB validation, and `X-Manage-Token` ownership enforcement
- Constant-time TA auth, `X-TA-Token` verification, and mocked Gemini synthesis
- Standardized HTTP/validation/500 error envelopes, XSS escaping, SQLite pragmas/indexes, and WCAG UI checks

---

## Render Deployment

DoubtHub runs cleanly on low-memory cloud instances (including Render free/starter tiers) by offloading embeddings and LLM calls to the Gemini API:

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment Variables:** Set `GEMINI_API_KEY` (and optionally `TA_EMAIL`, `TA_PASSWORD`, `TA_SESSION_TOKEN`) in the Render dashboard.

