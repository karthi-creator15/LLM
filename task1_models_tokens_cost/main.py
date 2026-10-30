from pathlib import Path
import json
import tiktoken

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

SAMPLES = {
    "sentence": "Large language models can understand and generate human language.",
    "paragraph": ("A foundation model is trained on broad data and can be adapted to many tasks. "
                  "Businesses use these models for summarisation, classification, extraction and support."),
    "page": ("Artificial intelligence is changing how people interact with software. " * 95),
    "json": json.dumps({
        "candidate_name": "Ananya",
        "skills": ["Python", "SQL", "Machine Learning", "Communication"],
        "years_experience": 2,
        "education": [{"degree": "BSc Computer Science", "year": 2024}]
    }, ensure_ascii=False),
    "tamil": "வணக்கம்! பெரிய மொழி மாதிரிகள் உரையைப் புரிந்துகொண்டு பதில்களை உருவாக்குகின்றன."
}

def estimate_cost(input_tokens, output_tokens, price_in, price_out):
    """Prices are USD per 1 million tokens."""
    return (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out

def main():
    enc = tiktoken.get_encoding("cl100k_base")
    rows = []
    for name, value in SAMPLES.items():
        tokens = len(enc.encode(value))
        words = len(value.split())
        rows.append({
            "text": name, "words": words, "tokens": tokens,
            "words_per_token": round(words / tokens, 3) if tokens else 0,
            "fits_8k": tokens <= 8192, "fits_128k": tokens <= 128000
        })
    print("TOKEN COUNTS (cl100k_base)")
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    (OUT / "token_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    chats_per_day = 1000
    input_per_chat, output_per_chat = 1500, 400
    monthly_input = chats_per_day * input_per_chat * 30
    monthly_output = chats_per_day * output_per_chat * 30

    # Replace these illustrative rates with the exact selected model's current official rates.
    price_in, price_out = 0.50, 1.50  # USD per 1M tokens; illustrative only
    cost = estimate_cost(monthly_input, monthly_output, price_in, price_out)
    print("\nMONTHLY ESTIMATE")
    print(f"Chats/month: {chats_per_day * 30:,}")
    print(f"Input tokens: {monthly_input:,}; output tokens: {monthly_output:,}")
    print(f"Illustrative rates: ${price_in}/1M input, ${price_out}/1M output")
    print(f"Estimated monthly cost: ${cost:.2f} USD")
    print("Update rates from the official pricing page for the exact model before submission.")

    print("\nMODEL RECOMMENDATIONS")
    print("Public support bot: choose a hosted GPT/Claude/Gemini model after a ticket-specific evaluation.")
    print("Reasons: managed scaling, strong instruction following, monitoring and simpler operations.")
    print("Use retrieval for company policy and human handoff for uncertain/high-impact cases.")
    print("Private on-prem tool: choose an appropriately licensed Llama model.")
    print("Reasons: self-hosting, data locality, control over deployment and customization.")
    print("Validate hardware, latency, model license, security and quality before deployment.")

if __name__ == "__main__":
    main()
