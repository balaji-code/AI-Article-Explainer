import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def extract_text_from_url(url: str) -> str:
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch URL: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")

    paragraph_text = " ".join(
        p.get_text() for p in paragraphs if len(p.get_text()) > 50
    )

    clean_text = " ".join(paragraph_text.split())
    return clean_text


def extract_text_from_pdf(file) -> str:
    """
    Extract clean text from a text-based PDF file.
    Note: This does NOT support scanned/image PDFs (no OCR).
    """
    reader = PdfReader(file)

    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    if not text_parts:
        raise RuntimeError("No extractable text found in PDF (possibly scanned).")

    text = " ".join(text_parts)
    clean_text = " ".join(text.split())
    return clean_text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0

    return chunks


def explain_text(chunks: list, query: str, k: int = 5) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embeddings.append(response.data[0].embedding)

    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    query_vector = np.array([query_embedding]).astype("float32")

    _, indices = index.search(query_vector, k)

    context = "\n\n".join(chunks[idx] for idx in indices[0])

    prompt = f"""
You are an expert explainer.

Using ONLY the information in the context below, explain the topic clearly.
Do not add external knowledge.
If something is not in the context, say so.

Context:
{context}

Explanation:
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
