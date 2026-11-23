
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

# --------------------------------------------------
# 1. ENVIRONMENT CONFIGURATION
# --------------------------------------------------

ENV_PATH = Path(__file__).resolve().parent / ".env"

if not ENV_PATH.exists():
    raise FileNotFoundError(f".env file not found: {ENV_PATH}")

# IMPORTANT: Override stale system environment variables
load_dotenv(dotenv_path=ENV_PATH, override=True)


def required_env(name):
    value = os.getenv(name, "").strip()

    # Remove accidental surrounding quotes
    value = value.strip('"').strip("'")

    if not value:
        raise RuntimeError(f"Missing {name} in .env")

    return value


QDRANT_URL = required_env("QDRANT_URL")
QDRANT_API_KEY = required_env("QDRANT_API_KEY")
GROQ_API_KEY = required_env("GROQ_API_KEY")

# Validate URL without modifying it
parsed = urlsplit(QDRANT_URL)

if parsed.scheme != "https" or not parsed.hostname:
    raise ValueError(
        "Invalid QDRANT_URL. Use the HTTPS REST endpoint "
        "provided by your Qdrant Cloud dashboard."
    )

COLLECTION_NAME = "knowledge"
MODEL_NAME = "all-MiniLM-L6-v2"

print("Environment loaded successfully.")
print("Qdrant host:", parsed.hostname)
print("API key present:", bool(QDRANT_API_KEY))
print("API key length:", len(QDRANT_API_KEY))


# --------------------------------------------------
# 2. DOCUMENTS
# --------------------------------------------------

documents = [
    "Employees receive 24 days of paid annual leave (vacation) per year.",
    "Employees receive 12 days of sick leave per year.",
    "Employees receive medical health insurance covering hospitalization and medical expenses.",
    "Employees receive dental and vision insurance as part of their healthcare benefits.",
    "Employees receive Rs 3000 per month for gym membership reimbursement.",
    "Employees can claim Rs 2000 per month for home internet expenses.",
    "Employees receive a company laptop and necessary accessories for work.",
    "Employees can work remotely on Monday and Friday.",
    "Employees are required to work from the office on Tuesday, Wednesday and Thursday.",
    "Employees receive free lunch, snacks and beverages at the office.",
    "Employees receive performance-based bonuses depending on individual and company performance.",
    "Employees are eligible for annual salary increments based on performance.",
    "Employees can participate in the company's Employee Stock Option Plan (ESOP).",
    "Employees receive 16 weeks of paid maternity leave, subject to company policy and applicable laws.",
    "Employees receive paid paternity leave according to company policy.",
    "Employees can access free online courses and professional certification programs.",
    "Employees receive financial assistance for attending technical conferences and workshops.",
    "Employees receive Rs 5000 per year for professional learning and development.",
    "Employees receive relocation assistance when moving to another city for work.",
    "Employees receive transportation allowance for commuting to the office.",
    "Employees have a 90-day notice period before leaving the company.",
    "Employees are eligible for gratuity according to applicable laws and company policy.",
    "Employees receive Provident Fund (PF) contributions as per applicable regulations.",
    "Employees can access mental health counselling and wellness programs.",
    "Employees receive 10 paid public holidays every year.",
    "Employees can take flexible working hours with manager approval.",
    "Employees are eligible for a yearly health checkup sponsored by the company.",
    "Employees receive company-sponsored team outings and annual celebrations.",
    "Employees can claim reimbursement for approved work-related expenses.",
    "Employees receive overtime compensation according to company policy and applicable laws.",
]


# --------------------------------------------------
# 3. CONNECT TO QDRANT
# --------------------------------------------------

print("\nConnecting to Qdrant Cloud...")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30,
    check_compatibility=False,
)

try:
    collections = client.get_collections()
    print("Qdrant connection successful!")

    print(
        "Available collections:",
        [c.name for c in collections.collections]
    )

except Exception as exc:
    print("\nQDRANT CONNECTION FAILED")
    print("Error:", str(exc))

    if getattr(exc, "status_code", None) == 403:
        raise RuntimeError(
            "\nHTTP 403: Qdrant rejected the request.\n"
            "1. Generate a fresh API key in your cluster dashboard.\n"
            "2. Verify that the URL and key belong to the same cluster.\n"
            "3. Check cluster IP restrictions and API key permissions.\n"
        ) from exc

    if getattr(exc, "status_code", None) == 401:
        raise RuntimeError(
            "HTTP 401: Invalid or missing API credentials."
        ) from exc

    raise


# --------------------------------------------------
# 4. LOAD EMBEDDING MODEL
# --------------------------------------------------

print(f"\nLoading embedding model: {MODEL_NAME}")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(MODEL_NAME)

embedding_size = model.get_sentence_embedding_dimension()

if not embedding_size:
    raise RuntimeError("Could not determine embedding dimensions.")

print("Embedding model ready.")
print("Vector dimensions:", embedding_size)


# --------------------------------------------------
# 5. CREATE OR VALIDATE COLLECTION
# --------------------------------------------------

if client.collection_exists(COLLECTION_NAME):

    collection_info = client.get_collection(COLLECTION_NAME)
    vectors = collection_info.config.params.vectors

    if not isinstance(vectors, VectorParams):
        raise RuntimeError(
            "Existing collection uses named vectors. "
            "Resolve its configuration manually."
        )

    if (
        vectors.size != embedding_size
        or vectors.distance != Distance.COSINE
    ):
        raise RuntimeError(
            "Existing collection has incompatible vector settings. "
            "No collection was deleted or modified."
        )

    print(f"\nUsing existing collection: {COLLECTION_NAME}")

else:

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embedding_size,
            distance=Distance.COSINE,
        ),
    )

    print(f"\nCreated collection: {COLLECTION_NAME}")


# --------------------------------------------------
# 6. GENERATE DOCUMENT EMBEDDINGS
# --------------------------------------------------

print("\nGenerating document embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True,
    normalize_embeddings=True,
)

points = [
    PointStruct(
        id=index + 1,
        vector=embedding.tolist(),
        payload={"text": document},
    )
    for index, (document, embedding)
    in enumerate(zip(documents, embeddings))
]


# --------------------------------------------------
# 7. UPLOAD TO QDRANT
# --------------------------------------------------

print("\nUploading documents to Qdrant...")

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
    wait=True,
)

print(f"Successfully uploaded {len(points)} documents!")


# --------------------------------------------------
# 8. SEMANTIC SEARCH
# --------------------------------------------------

def search(query: str, top_k: int = 3):

    query_vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return response.points


# --------------------------------------------------
# 9. GROQ LLM
# --------------------------------------------------


GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
groq_client = Groq(api_key=GROQ_API_KEY)


def ask_llm(question: str, context: str) -> str:

    prompt = f"""
You are a helpful company policy assistant.

Answer the question using ONLY the information
provided in the context.

Rules:
1. Do not invent or assume information.
2. Give a direct and concise answer.
3. If the answer is unavailable, say:
   "I don't know based on the provided information."

Context:
{context}

Question:
{question}

Answer:
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
        max_tokens=200,
    )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError("Groq returned an empty response.")

    return answer.strip()



# --------------------------------------------------
# 10. TEST COMPLETE RAG PIPELINE
# --------------------------------------------------

question = "How much vacation do I get?"

print("\nSearching knowledge base...")

results = search(question, top_k=3)

context_documents = [
    result.payload["text"]
    for result in results
    if result.payload
    and isinstance(result.payload.get("text"), str)
]

if not context_documents:
    raise RuntimeError("No relevant documents found.")

context = "\n".join(context_documents)

print("\nRetrieved Documents:")

for result in results:

    if result.payload and result.payload.get("text"):

        print(f"\nScore: {result.score:.4f}")
        print(result.payload["text"])


print("\n" + "=" * 50)
print("FINAL RAG ANSWER")
print("=" * 50)

answer = ask_llm(question, context)

print(answer)

print("\nRAG PIPELINE COMPLETED SUCCESSFULLY!")
