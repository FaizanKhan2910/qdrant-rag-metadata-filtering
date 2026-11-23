# Qdrant RAG with Metadata Filtering

A Retrieval-Augmented Generation (RAG) pipeline built using **Python, Qdrant Vector Database, Sentence Transformers, and Groq LLM**.

This project demonstrates how AI applications can retrieve contextually relevant information using semantic search and metadata filtering before generating answers with an LLM.

## 🚀 Features

- **Semantic Search:** Retrieve documents based on contextual meaning rather than exact keyword matching.
- **Vector Embeddings:** Generate 384-dimensional embeddings using `all-MiniLM-L6-v2`.
- **Vector Database:** Store and retrieve embeddings using Qdrant Cloud.
- **Metadata Filtering:** Filter documents using Qdrant payload conditions.
- **Cosine Similarity:** Find semantically similar documents.
- **LLM Integration:** Generate answers using Groq.
- **Context-Grounded Responses:** Answer questions using retrieved documents and provide a fallback when information is unavailable.

## 🛠️ Tech Stack

- Python
- Qdrant Cloud
- Sentence Transformers
- all-MiniLM-L6-v2
- Groq API
- Vector Embeddings
- Retrieval-Augmented Generation (RAG)

## 📂 Project Structure

```text
Vector/
│
├── qdrant.py
├── qdrantfull.py
├── knowledge.json
├── requirements.txt
├── .gitignore
├── README.md
└── .env
```

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/FaizanKhan2910/qdrant-rag-metadata-filtering.git
cd qdrant-rag-metadata-filtering
```

**2. Create a virtual environment**

```bash
python3 -m venv .venv
```

**3. Activate the environment**

Mac/Linux:

```bash
source .venv/bin/activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Configure environment variables**

Create a `.env` file:

```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
```

**6. Run the project**

```bash
python3 qdrantfull.py
```

## 🔍 How It Works

1. Load documents from `knowledge.json`.
2. Convert documents into vector embeddings using Sentence Transformers.
3. Store vectors and metadata in Qdrant.
4. Convert the user's query into an embedding.
5. Apply metadata filters and perform semantic search.
6. Retrieve relevant documents from Qdrant.
7. Pass retrieved context to the Groq LLM.
8. Generate a context-grounded response.

## 🧠 Key Concepts Learned

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Databases
- Sentence Transformers
- Vector Embeddings
- Cosine Similarity
- Metadata Filtering
- LLM Integration

## 👨‍💻 Author

**Faizan Khan**

GitHub: [@FaizanKhan2910](https://github.com/FaizanKhan2910)
