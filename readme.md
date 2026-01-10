# AI Article Explainer (RAG)

This project is a minimal, from-scratch implementation of a **grounded Retrieval-Augmented Generation (RAG) pipeline**.

It demonstrates how to build a correct RAG system end-to-end without frameworks, abstractions, or agentic layers.

## What it does
- Accepts a **live article URL** at runtime
- Fetches and parses the webpage content
- Cleans and extracts human-readable text
- Chunks the text deterministically with overlap
- Embeds chunks into a vector space
- Retrieves the most relevant chunks for a query
- Generates a **grounded explanation using only retrieved context**

## What this project deliberately avoids
- No frameworks (LangChain, LlamaIndex, etc.)
- No agents or tool orchestration
- No persistent databases
- No prompt stuffing with full documents
- No hidden abstractions

Everything is explicit and inspectable.

## Why this exists
Most RAG examples fail silently due to:
- poor ingestion
- noisy chunking
- unverified retrieval
- unconstrained generation

This project exists to make each stage observable and debuggable, so failures are attributable to specific pipeline steps.

## Current status
- URL ingestion: ✅
- HTML parsing and cleaning: ✅
- Deterministic chunking: ✅
- Embeddings creation: ✅
- Vector retrieval (FAISS, in-memory): ✅
- Grounded LLM generation: ✅

## Architecture overview
1. URL → raw HTML
2. HTML → clean text
3. Text → fixed-size overlapping chunks
4. Chunks → embeddings
5. Embeddings → vector index (FAISS)
6. Query → top-k retrieved chunks
7. Retrieved chunks → constrained LLM explanation

## Tech stack
- Python
- requests
- BeautifulSoup
- NumPy
- FAISS (in-memory)
- OpenAI API (embeddings + generation)

## Notes
- API keys are loaded via `.env` and are never committed
- The vector index is rebuilt per run (no persistence by design)
- Wikipedia articles are used as stress-test inputs, not as a dependency

This repository represents a **minimal correct RAG baseline**.