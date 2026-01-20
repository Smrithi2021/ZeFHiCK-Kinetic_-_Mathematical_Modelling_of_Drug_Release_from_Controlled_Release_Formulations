from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pathlib import Path
import subprocess

# Load RAG DB
DB_DIR = Path("rag_db")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=str(DB_DIR),
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 3})


def ask_gemma(prompt: str) -> str:
    result = subprocess.run(
        [
            "C:\\Users\\Smrithi\\AppData\\Local\\Programs\\Ollama\\ollama.exe",
            "run",
            "gemma:2b",
        ],
        input=prompt,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def explain_subtype(subtype: str) -> str:
    docs = retriever.invoke(
        f"PAM50 {subtype} breast cancer biology"
    )

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are an oncology bioinformatics assistant.

Predicted PAM50 subtype: {subtype}

Scientific context:
{context}

Explain:
1. What this subtype means biologically
2. Why it matters clinically
3. How it relates to gene expression
"""

    return ask_gemma(prompt)
