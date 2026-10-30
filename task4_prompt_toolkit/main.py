import json, os, re, time
from pathlib import Path
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, Field

load_dotenv()
BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

class ResumeOutput(BaseModel):
    name: str | None
    skills: list[str] = Field(default_factory=list)
    years_experience: float | None

def generate(prompt, temperature=0, top_p=0.95, format_json=False):
    body = {"model": MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "top_p": top_p}}
    if format_json:
        body["format"] = "json"
    r = requests.post(f"{BASE}/api/generate", json=body, timeout=180)
    r.raise_for_status()
    return r.json()["response"].strip()

def parse_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group())
        raise

def classify(prompt):
    try:
        obj = parse_json(generate(prompt))
        return obj.get("category"), obj.get("priority"), obj
    except Exception:
        return None, None, None

def main():
    tickets = json.loads((ROOT / "data/tickets.json").read_text(encoding="utf-8"))
    resumes = json.loads((ROOT / "data/resumes.json").read_text(encoding="utf-8"))
    prompt_results = {}
    for version in ("v1", "v2", "v3"):
        template = (ROOT / f"prompts/classification_{version}.txt").read_text(encoding="utf-8")
        correct = 0
        predictions = []
        for item in tickets:
            prompt = template.format(ticket=item["text"])
            category, priority, raw = classify(prompt)
            ok = category == item["category"] and priority == item["priority"]
            correct += int(ok)
            predictions.append({"id": item["id"], "expected_category": item["category"],
                                "expected_priority": item["priority"], "predicted": raw, "correct": ok})
        prompt_results[version] = {"correct": correct, "total": len(tickets),
                                   "accuracy": correct / len(tickets), "predictions": predictions}
        print(version, f"{correct}/{len(tickets)} ({correct/len(tickets):.1%})")

    # Chain-of-thought comparison: score final answers only; do not request or store private reasoning.
    word_problems = [
        ("A pen costs ₹12. What do 4 pens cost?", "48"),
        ("A bus has 30 passengers. 8 get off and 5 get on. How many remain?", "27"),
        ("A box has 6 rows of 7 items. How many items?", "42"),
        ("Ravi has 50. He spends 18 and then 12. How much remains?", "20"),
        ("Three friends share 81 equally. How much does each get?", "27")
    ]
    cot = []
    for question, expected in word_problems:
        plain = generate(question + " Return only the final numeric answer.")
        step = generate(question + " Solve carefully, checking the arithmetic. Return only the final numeric answer.")
        cot.append({"question": question, "expected": expected, "without_step_prompt": plain,
                    "with_step_prompt": step,
                    "plain_correct": expected in plain, "step_prompt_correct": expected in step})

    creative_prompt = "Write a two-sentence imaginative description of a city floating above the ocean."
    sampling = {}
    for temp in (0, 0.7, 1.2):
        for top_p in (0.3, 0.95):
            outputs = [generate(creative_prompt, temperature=temp, top_p=top_p) for _ in range(5)]
            sampling[f"temperature_{temp}_top_p_{top_p}"] = outputs

    extraction = {"plain": [], "json_mode": []}
    invalid = {"plain": 0, "json_mode": 0}
    for resume in resumes:
        prompt = ("Extract name, skills and years_experience from this resume. "
                  "Return JSON only. Use null if unknown. Resume: " + resume["text"])
        for mode, use_json in (("plain", False), ("json_mode", True)):
            try:
                raw = generate(prompt, format_json=use_json)
                parsed = parse_json(raw)
                validated = ResumeOutput.model_validate(parsed).model_dump()
                extraction[mode].append({"raw": raw, "validated": validated,
                                         "expected": resume["expected"]})
            except (Exception, ValidationError) as exc:
                invalid[mode] += 1
                extraction[mode].append({"error": str(exc), "expected": resume["expected"]})

    bad_prompt_rewrites = [
        {"bad": "Tell me about this ticket.", "improved": "Classify the ticket into Billing, Technical or Account. Assign Low, Medium or High priority. Return only JSON with category and priority. Ticket: {ticket}"},
        {"bad": "Extract details from this CV.", "improved": "Extract only name, skills array and years_experience. Never infer missing facts; use null for missing scalar values. Return valid JSON matching the requested schema. Resume: {resume}"},
        {"bad": "Do the right thing with this customer issue.", "improved": "Act as a support triage classifier. Use only the supplied text, choose one allowed category and priority, and return the exact JSON schema without commentary. Ticket: {ticket}"}
    ]
    payload = {"classification": prompt_results, "chain_of_thought_prompt_test": cot,
               "sampling_experiment": sampling, "resume_extraction": extraction,
               "invalid_json_or_schema_count": invalid, "bad_prompt_rewrites": bad_prompt_rewrites,
               "notes": ["Scores are measured from this local model; do not hardcode example scores.",
                         "The chain-of-thought experiment evaluates final answer accuracy only.",
                         "JSON mode constrains format but Pydantic validation is still required."]}
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nResume invalid counts:", invalid)
    print("Full outputs saved to", OUT / "results.json")

if __name__ == "__main__":
    main()
