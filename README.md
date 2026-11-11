# DoubtHub 💡

An intelligent academic doubt-resolution platform that eliminates redundant questions through **hybrid AI clustering**, surfaces instant peer solutions, and connects students with verified peer mentors.

🔗 **Live Demo:** [doubt-hub.onrender.com](https://doubt-hub.onrender.com)

---

## 🚀 Key Features

* **Hybrid Doubt Clustering**: Automatically groups semantically similar student doubts into shared question clusters using embeddings and cloud LLM APIs.
* **Instant Peer Solutions**: Highlights existing solutions and student discussions so users get answers immediately without waiting for instructor replies.
* **Persistent Knowledge Base**: Stores questions, threads, and cluster states locally via SQLite (`doubts.db`).
* **Clean Interactive UI**: Fast, responsive web frontend powered by FastAPI and modern UI templates.
* **24/7 Cloud Ready**: Migrated from local models to cloud API endpoints for continuous cloud deployment on platforms like Render.

---

## 🛠️ Tech Stack

* **Backend**: Python, FastAPI / Uvicorn
* **Clustering & AI**: Semantic embeddings & LLM cloud API (`cluster.py`)
* **Database**: SQLite / SQLModel / SQLAlchemy (`models.py`, `doubts.db`)
* **Frontend**: HTML5, Modern CSS / Tailwind, JavaScript (`static/`)

---

## 📁 Repository Structure

```text
doubt-hub/
├── static/           # Frontend assets, HTML templates, styles, and client scripts
├── cluster.py        # Semantic clustering logic and cloud API integration
├── main.py           # FastAPI application routes and server entry point
├── models.py         # Database schema definitions and ORM models
├── doubts.db         # SQLite database storing doubts, responses, and clusters
├── requirements.txt  # Project dependencies
└── .gitignore        # Ignored files and environment rules
