import json
import os
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv()
BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

def ask(prompt, temperature=0):
    response = requests.post(
        f"{BASE}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False,
              "options": {"temperature": temperature}},
        timeout=180
    )
    response.raise_for_status()
    return response.json()["response"].strip()

def main():
    tests = [
        ("Summarisation", "Summarise in one sentence: Our support team received 120 tickets this week. "
         "Billing was 45%, technical 35%, account access 20%. Average first response time was 2 hours."),
        ("Classification", "Classify this ticket as Billing, Technical, or Account. Return only the label: "
         "I was charged twice for the same subscription."),
        ("Extraction", "Extract product, issue and requested_action as JSON from: "
         "'The mobile app crashes on login. Please fix my access.'"),
        ("Translation", "Translate into Tamil: Your payment was received successfully."),
        ("Code generation", "Write a Python function named is_even(n) that returns True for even integers.")
    ]
    results = []
    for name, prompt in tests:
        try:
            result = ask(prompt)
            rating = int(input(f"Rate {name} result 1-5: "))
        except Exception as exc:
            result, rating = f"ERROR: {exc}", 1
        results.append({"test": name, "prompt": prompt, "result": result, "rating": rating})

    cutoff_prompt = ("Who won the 2026 Nobel Prize in Physics? If you do not know, say so. "
                     "Do not guess. This is a test of current knowledge.")
    cutoff = ask(cutoff_prompt)
    results.append({"test": "Knowledge cutoff", "prompt": cutoff_prompt, "result": cutoff, "rating": None})

    invented = ("Summarise the findings of the fictional paper 'Quantum Sentiment Alignment in "
                "Neural Orchids' by Dr. Elara Voss (2024).")
    guarded = invented + " Say 'I don't know' if unsure. Do not invent a paper or author."
    hallucination = {
        "without_uncertainty_instruction": ask(invented),
        "with_uncertainty_instruction": ask(guarded)
    }

    filler = "This is ordinary context text with no special information. " * 250
    long_text = filler + "\nSECRET FACT: The project code is ORCHID-4729.\n" + filler
    context_prompt = long_text + "\nWhat is the exact project code? Return only the code."
    context_result = ask(context_prompt)

    arithmetic_prompt = ("A shop sells 3 notebooks for ₹45 each and 2 pens for ₹15 each. "
                         "A customer pays ₹200. How much change is due? Give the answer.")
    arithmetic_result = ask(arithmetic_prompt)
    logic_prompt = ("All bloops are razzies. All razzies are lazzies. Are all bloops lazzies? "
                    "Answer yes or no and give a brief reason.")
    logic_result = ask(logic_prompt)

    safeguards = [
        "Ground answers in approved company documents using retrieval (RAG) and cite retrieved sources.",
        "Validate structured output with JSON Schema or Pydantic; reject and retry invalid output.",
        "Use deterministic business rules or calculators for arithmetic, pricing and eligibility.",
        "Add confidence/uncertainty thresholds, refusal behavior and human escalation for risky cases.",
        "Log, monitor and regularly test hallucinations, bias, prompt injection and data leakage."
    ]
    payload = {"capability_tests": results, "cutoff_test": cutoff,
               "hallucination_comparison": hallucination,
               "context_test": {"expected": "ORCHID-4729", "actual": context_result},
               "reasoning_tests": {"arithmetic_prompt": arithmetic_prompt,
                                   "arithmetic_model_answer": arithmetic_result,
                                   "verified_arithmetic": "₹200 - (3*₹45 + 2*₹15) = ₹35",
                                   "logic_prompt": logic_prompt,
                                   "logic_model_answer": logic_result,
                                   "verified_logic": "Yes; by transitivity."},
               "business_safeguards": safeguards}
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("\nManually verify every model response and write your own ratings in the submission.")

if __name__ == "__main__":
    main()
