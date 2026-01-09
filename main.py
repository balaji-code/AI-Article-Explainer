import requests
from bs4 import BeautifulSoup

url = input("Enter article URL: ")

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

response = requests.get(url, headers=headers)

print("Status code:", response.status_code)

if response.status_code != 200:
    print("Request failed")
    exit()

soup = BeautifulSoup(response.text, "html.parser")

paragraphs = soup.find_all("p")

paragraph_text = " ".join(
    p.get_text() for p in paragraphs if len(p.get_text()) > 50
)

clean_text = " ".join(paragraph_text.split())

print(clean_text[:3000])

def chunk_text(text, chunk_size=500, overlap=100):
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

chunks = chunk_text(clean_text)
print("Average chunk length:",
      sum(len(c) for c in chunks) // len(chunks))
print(f"Total chunks created: {len(chunks)}")
print("\n--- First chunk ---\n")
print(chunks[0])