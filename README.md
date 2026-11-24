# ☁️ AWS RAG Chatbot

A retrieval-augmented generation (RAG) chatbot that answers questions about AWS services using official AWS documentation.

## Architecture

```
User Question
      │
      ▼
┌─────────────┐    ┌──────────────────┐    ┌────────────────┐
│  Streamlit   │───▶│  rag_chain.py    │───▶│  Groq LLM      │
│  Chat UI     │    │  retrieve()      │    │  gpt-oss-120b  │
│  (app.py)    │    │  stream_answer() │    └────────────────┘
└─────────────┘    └───────┬──────────┘
                           │
                   ┌───────▼──────────┐
                   │  FAISS Index     │
                   │  5000 chunks     │
                   │  384-dim vectors │
                   └──────────────────┘
```

## Tech Stack

| Component       | Technology                                      |
|----------------|-------------------------------------------------|
| Embeddings     | `all-MiniLM-L6-v2` (local, CPU)                |
| Vector Store   | FAISS `IndexFlatL2` (local file)                |
| LLM            | `openai/gpt-oss-120b` via Groq                 |
| UI             | Streamlit with streaming + chat history         |
| Evaluation     | RAGAS (Faithfulness + Context Precision)         |
| Package Mgr    | `uv`                                            |
| Data Source    | [AWS Public PDF Chunks](https://huggingface.co/datasets/semihk1/aws-public-pdf-chunked-dataset) |

## Setup

### 1. Clone & install

```bash
git clone <repo-url>
cd rag-project
uv sync
```

### 2. Set your Groq API key

Get a free key at [console.groq.com/keys](https://console.groq.com/keys), then:

```bash
# Option A: .streamlit/secrets.toml (recommended for local dev)
mkdir -p .streamlit
echo 'GROQ_API_KEY = "gsk_..."' > .streamlit/secrets.toml

# Option B: environment variable
export GROQ_API_KEY="gsk_..."
```

### 3. Build the FAISS index (if needed)

The index files (`aws_index.faiss`, `metadata.json`) are not tracked in git.
If they don't exist, run the ingestion notebook:

```bash
uv run jupyter nbconvert --to notebook --execute ingest.ipynb
```

> ⚠️ This downloads ~100k chunks from HuggingFace and encodes 5000 of them. Takes ~2 minutes on CPU.

### 4. Run the chatbot

```bash
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Evaluation

Run the RAGAS evaluation harness:

```bash
# Smoke test (3 questions)
uv run python eval.py --limit 3

# Full evaluation (50 questions)
uv run python eval.py --limit 50
```

Results are saved to `eval_results.csv`. Metrics:
- **Faithfulness** – Is the answer grounded in the retrieved context?
- **Context Precision** – Are the retrieved chunks relevant to the question?

## Project Structure

```
├── app.py              # Streamlit chat UI
├── rag_chain.py        # Core RAG pipeline (retrieve, answer, stream)
├── eval.py             # RAGAS evaluation harness
├── ingest.ipynb        # Data ingestion & FAISS index builder
├── aws_index.faiss     # Pre-built FAISS index (not in git)
├── metadata.json       # Chunk texts + sources (not in git)
├── pyproject.toml      # Project config & dependencies
├── uv.lock             # Locked dependency versions
├── requirements.txt    # Pip-compatible deps (for deployment)
└── .streamlit/
    └── secrets.toml    # API key config (not in git)
```

## License

MIT
