# Video Search and Summarization

A fork of [NVIDIA-AI-Blueprints/video-search-and-summarization](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) — an AI-powered pipeline for searching and summarizing video content using NVIDIA's vision and language models.

## Overview

This project enables users to:
- **Ingest** video files and extract frames, audio, and metadata
- **Index** video content using multimodal embeddings for semantic search
- **Search** across video libraries using natural language queries
- **Summarize** video segments with AI-generated descriptions

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Video Search & Summarization          │
├──────────────┬──────────────┬──────────────┬────────────┤
│   Ingestion  │   Indexing   │    Search    │ Summarize  │
│   Pipeline   │   Service    │    API       │  Service   │
├──────────────┴──────────────┴──────────────┴────────────┤
│              NVIDIA NIM / NeMo Microservices             │
│         (VLM, Embeddings, LLM, ASR)                     │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU (A100, H100, or RTX 3090/4090 recommended)
- NVIDIA Container Toolkit
- NVIDIA API Key (for NIM microservices)
- Python 3.10+

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/video-search-and-summarization.git
cd video-search-and-summarization
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your NVIDIA API key and configuration
```

### 3. Launch Services

```bash
docker compose up -d
```

### 4. Access the UI

Open your browser and navigate to `http://localhost:3000`

## Configuration

Key environment variables:

| Variable | Description | Default |
|---|---|---|
| `NVIDIA_API_KEY` | NVIDIA NIM API key | required |
| `VLM_MODEL` | Vision-language model to use | `nvidia/vila` |
| `EMBEDDING_MODEL` | Embedding model for indexing | `nvidia/nv-embedqa-e5-v5` |
| `LLM_MODEL` | LLM for summarization | `meta/llama-3.1-70b-instruct` |
| `VECTOR_DB_URL` | Milvus/pgvector connection URL | `localhost:19530` |
| `FRAME_EXTRACTION_FPS` | Frames per second to extract | `2` |

> **Personal note:** I bumped `FRAME_EXTRACTION_FPS` from `1` to `2` — found that 1 fps misses a lot of fast-moving content in the videos I'm working with. May increase further depending on GPU memory availability.

## Development

### Setup Local Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

This project uses `black` for formatting and `ruff` for linting:

```bash
black src/
ruff check src/
```

## Troubleshooting

**Services fail to start:** Run `docker compose logs -f` to check for missing env vars or port conflicts. Make sure `NVIDIA_API_KEY` is set in `.env`.

**Out of GPU memory during ingestion:** Lower `FRAME_EXTRACTION_FPS` or reduce batch size in `.env`. On my RTX 3090, `FRAME_EXTRACTION_FPS=2` works fine but 3+ causes OOM errors with the default VLM.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and review our [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md) before submitting changes.
