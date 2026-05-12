# Assignment 2 Starter Code: Shakespeare-Aware SLM/RAG System

This starter code provides a minimal scaffold for Assignment 2. It is **not** a complete solution. 
Your group must complete, modify, justify, evaluate, and document your own system.

## Expected Dataset Placement

Place the three provided play files in:

```text
data/processed/
  hamlet.json
  macbeth.json
  romeo_and_juliet.json
```

Each file is expected to contain structured records or scene chunks. You may adapt the loader if your dataset format differs.

## Suggested Workflow

1. Load the structured Shakespeare dataset.
2. Create retrieval chunks.
3. Generate embeddings.
4. Build a retrieval mechanism.
5. Retrieve relevant passages for a query.
6. Build a RAG prompt using retrieved passages.
7. Generate an answer using a selected language model or hosted API.
8. Evaluate baseline and RAG systems using instructor-provided and group-designed questions.

## Quick Start

Create a Python environment and install minimal dependencies:

```bash
pip install -r requirements.txt
```

Run a simple retrieval test:

```bash
python src/build_index.py
```

Run the chatbot scaffold:

```bash
python src/rag_chatbot.py
```

Run the evaluation scaffold:

```bash
python src/evaluate.py
```

## What You Must Add

You must add or complete:

- dataset loading adapted to the provided dataset;
- a justified chunking strategy;
- a working embedding model;
- a working retrieval method;
- a baseline system;
- a RAG-based system;
- a model/API interface for answer generation;
- evaluation results and failure analysis;
- documentation explaining how to run your system.

## Important

Do not submit the starter code unchanged. It is provided only to reduce setup friction.
