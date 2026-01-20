from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_DB_DIR = "vector_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings
)

def retrieve_context(query, k=3):
    docs = db.similarity_search(query, k=k)
    return "\n\n".join([d.page_content for d in docs])
