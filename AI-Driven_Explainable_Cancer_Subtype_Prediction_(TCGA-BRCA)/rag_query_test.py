from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load embeddings (same as build step)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector DB
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Query
query = "HER2 enriched breast cancer subtype biology"
docs = db.similarity_search(query, k=3)

for i, d in enumerate(docs, 1):
    print(f"\n--- Document {i} ---")
    print(d.page_content)
