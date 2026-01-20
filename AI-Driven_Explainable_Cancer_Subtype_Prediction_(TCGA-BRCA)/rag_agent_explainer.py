from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
import subprocess

# ---------- Load RAG database ----------
DB_DIR = Path("rag_db")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=str(DB_DIR),
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 3})


# ---------- LLM call ----------
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


# ---------- Agentic explanation ----------
def explain_prediction(predicted_subtype: str) -> str:
    # 1. Retrieve biology
    docs = retriever.invoke(
        f"PAM50 {predicted_subtype} breast cancer biology"
    )

    context = "\n\n".join(d.page_content for d in docs)

    # 2. Ask LLM to reason
    prompt = f"""
You are an oncology bioinformatics assistant.

Predicted molecular subtype: {predicted_subtype}

Use the following scientific context to explain:
{context}

Explain in simple language:
- What this subtype means biologically
- Why it matters clinically
"""

    return ask_gemma(prompt)


# ---------- Test ----------
if __name__ == "__main__":
    explanation = explain_prediction("HER2-enriched")
    print("\n🧠 AI Explanation:\n")
    print(explanation)
