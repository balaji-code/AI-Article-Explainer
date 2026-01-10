# Import required libraries
import requests  # For making HTTP requests to fetch article content
from openai import OpenAI  # OpenAI API client for generating embeddings
import faiss  # Facebook AI Similarity Search library (imported but not used in current code)
import numpy as np  # For numerical operations and array handling
import os  # For environment variable access
from dotenv import load_dotenv  # For loading environment variables from .env file
from bs4 import BeautifulSoup  # For parsing HTML content

# Get article URL from user input
url = input("Enter article URL: ")

# Set User-Agent header to mimic a real browser request
# This helps avoid being blocked by websites that reject automated requests
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

# Fetch the article content from the provided URL
response = requests.get(url, headers=headers)

# Display the HTTP status code to verify the request was successful
print("Status code:", response.status_code)

# Exit if the request failed (status code other than 200)
if response.status_code != 200:
    print("Request failed")
    exit()

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")

# Extract all paragraph elements from the HTML
paragraphs = soup.find_all("p")

# Join all paragraph texts into a single string
# Filter out paragraphs shorter than 50 characters to remove navigation/header elements
paragraph_text = " ".join(
    p.get_text() for p in paragraphs if len(p.get_text()) > 50
)

# Clean up the text by normalizing whitespace (replacing multiple spaces/tabs/newlines with single spaces)
clean_text = " ".join(paragraph_text.split())

# Display the first 3000 characters of the cleaned text for verification
print(clean_text[:3000])

def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks for processing.
    
    Args:
        text (str): The text to be chunked
        chunk_size (int): Maximum size of each chunk in characters (default: 500)
        overlap (int): Number of characters to overlap between chunks (default: 100)
    
    Returns:
        list: List of text chunks with specified size and overlap
    """
    chunks = []
    start = 0

    # Create chunks with overlapping windows to maintain context
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # Move start position back by overlap amount to create overlapping chunks
        start = end - overlap

        # Prevent negative start index
        if start < 0:
            start = 0

    return chunks

# Split the cleaned article text into chunks
chunks = chunk_text(clean_text)

# Display statistics about the created chunks
print("Average chunk length:",
      sum(len(c) for c in chunks) // len(chunks))
print(f"Total chunks created: {len(chunks)}")
print("\n--- First chunk ---\n")
print(chunks[0])

# Load environment variables from .env file
load_dotenv()
# Initialize OpenAI client (uses OPENAI_API_KEY from environment variables)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("Creating embeddings...")

# List to store embeddings for each chunk
embeddings = []

# Generate embeddings for each text chunk using OpenAI's embedding model
for i, chunk in enumerate(chunks):
    # Call OpenAI API to create embedding vector for the chunk
    response = client.embeddings.create(
        model="text-embedding-3-small",  # Using OpenAI's small, efficient embedding model
        input=chunk
    )
    # Extract the embedding vector from the response and add to list
    embeddings.append(response.data[0].embedding)

    # Print progress every 20 chunks to track processing status
    if i % 20 == 0:
        print(f"Embedded {i}/{len(chunks)} chunks")

# Convert list of embeddings to a NumPy array with float32 precision
# float32 is used for memory efficiency and compatibility with vector databases
embeddings = np.array(embeddings).astype("float32")

# Display the shape of the embeddings array (number of chunks, embedding dimensions)
print("Embeddings shape:", embeddings.shape)

# Get the number of dimensions in the embedding space
dimension = embeddings.shape[1]
# Create a flat L2 index for efficient nearest neighbor search
index = faiss.IndexFlatL2(dimension)
# Add the embeddings to the index
index.add(embeddings)
# Display the total number of vectors in the index
print("Total vectors in index:", index.ntotal)

# Define the query
query = "What is artificial intelligence?"

# Call OpenAI API to create embedding vector for the query
query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding

# Convert the query embedding to a NumPy array with float32 precision
query_vector = np.array([query_embedding]).astype("float32")

# Define the number of nearest neighbors to retrieve
k = 5
distances, indices = index.search(query_vector, k)
# Display the top retrieved chunks
print("\nTop retrieved chunks:\n")
# Display the top retrieved chunks
for idx in indices[0]:
    print("-----")
    print(chunks[idx][:500])

# Join the top retrieved chunks into a single string with newline separators
context = "\n\n".join(
    chunks[idx] for idx in indices[0]
)

# Define the prompt for the chat completion
prompt = f"""
You are an expert explainer.

Using ONLY the information in the context below, explain the topic clearly.
Do not add external knowledge.
If something is not in the context, say so.

Context:
{context}

Explanation:
"""

# Call OpenAI API to create a chat completion
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

# Display the final explanation
print("\n--- Final Explanation ---\n")
print(response.choices[0].message.content)