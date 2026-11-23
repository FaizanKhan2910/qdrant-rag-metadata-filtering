# ============================================================
# PART 1 — IMPORTS AND ENVIRONMENT
# ============================================================

import os
import json
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue, MatchAny, PayloadSchemaType
)
from sentence_transformers import SentenceTransformer
from groq import Groq

ENV_PATH = Path(__file__).resolve().parent / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

QDRANT_URL = os.getenv("QDRANT_URL", "").strip().strip('"').strip("'")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip().strip('"').strip("'")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")

if not QDRANT_URL or not QDRANT_API_KEY or not GROQ_API_KEY:
    raise RuntimeError("Missing environment variables in .env")

print("Environment loaded successfully.")
print("Qdrant URL:", QDRANT_URL)
print("Qdrant API key present:", bool(QDRANT_API_KEY))
# ============================================================
# PART 2 — CONNECT TO QDRANT
# ============================================================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30,
    check_compatibility=False,
)

try:
    collections = client.get_collections()
    print("Qdrant connection successful!")
    print("Collections:", [c.name for c in collections.collections])

except Exception as e:
    print("QDRANT ERROR:", repr(e))
    raise
# ============================================================
# PART 3 — CREATE QDRANT COLLECTION
# ============================================================

COLLECTION_NAME = "knowledge_filter"
EMBEDDING_SIZE = 384

if not client.collection_exists(COLLECTION_NAME):

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print(f"Created collection: {COLLECTION_NAME}")

else:
    print(f"Using existing collection: {COLLECTION_NAME}")


print(f"Vector size: {EMBEDDING_SIZE}")
print("Distance: COSINE")

client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD,
    wait=True,
)

print("Category payload index ready!")
# ============================================================
# PART 4 — LOAD OUR KNOWLEDGE
# ============================================================

with open("knowledge.json", "r", encoding="utf-8") as f:
    documents = json.load(f)


# ============================================================
# PART 5 — CREATE EMBEDDINGS
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2") #384

print("Embedding model ready!")
texts = [document["text"] for document in documents]


embeddings = model.encode(texts)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding size: {len(embeddings[0])}")


# ============================================================
# PART 6 — CREATE QDRANT POINTS
# ============================================================

points = []

for i in range(len(documents)):
    
    point = PointStruct(
        id=i + 1,
        vector=embeddings[i].tolist(),
        payload=documents[i]
    )

    points.append(point)


# ============================================================
# PART 7 — UPLOAD TO QDRANT
# ============================================================

client.upsert( #upload+insert
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"Uploaded {len(points)} documents to Qdrant!")


# ============================================================
# PART 8 — SEARCH QDRANT
# ============================================================

def search(query, top_k=3):

    # Convert the question into an embedding
    query_vector = model.encode(query).tolist()

    # Search Qdrant for similar vectors
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    return results

def search_with_filter(query, query_filter=None, top_k=3):


    query_vector = model.encode(query).tolist()


    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        query_filter=query_filter,
    ).points


    return results

reimbursement_filter = Filter(
    must=[
        FieldCondition(
            key="category",
            match=MatchValue(value="reimbursement")
        ), 
         
    ]
)

# ============================================================
# PART 9 — TEST SEARCH WITH FILTER
# ============================================================

query = "How do I claim reimbursement?"

results = search_with_filter(
    query,
    query_filter=reimbursement_filter,
    top_k=3
)

print("\nFiltered Search Results:")

for result in results:
    print(f"Score: {result.score:.3f}")
    print("Category:", result.payload["category"])
    print(result.payload["text"])
    print()
# ============================================================
# PART 10 — CONNECT TO GROQ
# ============================================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)
# ============================================================
# PART 11 — ASK THE LLM
# ============================================================

def ask_llm(question, context):

    if not context.strip():
        return "I don't know based on the provided information."

    prompt = f"""
You are an assistant answering questions about company policies.

STRICT RULES:
1. Answer using ONLY the provided context.
2. Do not use outside knowledge or make assumptions.
3. If the context does not directly answer the question,
   reply exactly:
   I don't know based on the provided information.
4. Do not confuse reimbursement amounts with reimbursement procedures.

Context:
{context}

Question:
{question}

Answer:
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# ============================================================
# PART 12 — COMPLETE RAG PIPELINE
# ============================================================

question = "How do I claim reimbursement?"

results = search_with_filter(
    question,
    query_filter=reimbursement_filter,
    top_k=3
)

# Extract text from the search results
context = "\n".join(
    result.payload["text"]
    for result in results
)


answer = ask_llm(question, context)


print("\nFinal Answer:")
print(answer)
client.close()