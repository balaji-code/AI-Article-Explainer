# AI Article Explainer (RAG)

This project is a minimal, from-scratch implementation of a Retrieval-Augmented Generation (RAG) pipeline.

## What it does
- Fetches an article from a live URL
- Extracts and cleans human-readable text
- Chunks the text deterministically
- (Next) embeds and retrieves relevant chunks for explanation

## Purpose
This project is built as a learning-by-building exercise:
- No frameworks
- No abstractions
- No agentic systems

The goal is to understand RAG mechanics end-to-end.

## Current status
- URL ingestion: ✅
- HTML parsing and cleaning: ✅
- Chunking: ✅
- Embeddings and retrieval: ⏳

## Tech stack
- Python
- requests
- BeautifulSoup
- FAISS (planned)
- OpenAI API (planned)