import subprocess

def ask_gemma(prompt: str) -> str:
    """
    Sends a prompt to Gemma via Ollama and returns the response
    """
    result = subprocess.run(
        ["F:\\Ollama\\ollama.exe", "run", "gemma:2b"],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()


if __name__ == "__main__":
    test_prompt = """
You are an oncology research assistant.
Explain HER2-enriched breast cancer subtype in simple terms.
"""
    print(ask_gemma(test_prompt))
