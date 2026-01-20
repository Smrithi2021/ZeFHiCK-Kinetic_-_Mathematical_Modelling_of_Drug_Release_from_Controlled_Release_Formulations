from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter



# -------- Paths --------
BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"
VECTOR_DB_DIR = BASE_DIR / "vector_db"


# -------- Load embedding model --------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------- Load knowledge files --------
texts = []

for file in KB_DIR.glob("*.txt"):
    content = file.read_text(encoding="utf-8").strip()
    if content:
        texts.append(content)
        print(f"📄 Loaded {file.name} ({len(content)} chars)")


if not texts:
    raise ValueError("❌ No knowledge text found in knowledge_base/*.txt")


# -------- Split into chunks --------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

documents = splitter.create_documents(texts)
print(f"✂️ Total chunks created: {len(documents)}")


# -------- Build vector DB --------
db = Chroma.from_documents(
    documents,
    embedding=embeddings,
    persist_directory=str(VECTOR_DB_DIR)
)

print("✅ Vector database successfully created")
