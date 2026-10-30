# Week 4 – LLM Fundamentals Assessment

A runnable, individual assessment project split into four tasks. The project uses **Ollama locally by default** for LLM experiments, so no paid API key is required. Task 1 uses `tiktoken` for token counting. Task 3 uses NumPy, Matplotlib and Hugging Face models.

## 1. Folder structure

```text
week4_llm_fundamentals/
├── task1_models_tokens_cost/
│   ├── main.py
│   └── model_comparison.md
├── task2_capabilities_limitations/
│   └── main.py
├── task3_transformer_concepts/
│   └── main.py
├── task4_prompt_toolkit/
│   ├── main.py
│   ├── prompts/
│   │   ├── classification_v1.txt
│   │   ├── classification_v2.txt
│   │   └── classification_v3.txt
│   └── data/
│       ├── tickets.json
│       └── resumes.json
├── requirements.txt
├── .env.example
└── .gitignore
```

## 2. Setup

Python 3.10–3.12 is recommended. Create a virtual environment:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install packages:
```bash
pip install -r requirements.txt
```

## 3. Ollama setup (local and free)

1. Install Ollama from https://ollama.com/download
2. Open a terminal and download a model:
   ```bash
   ollama pull llama3.2:3b
   ```
3. Start Ollama (the desktop app normally starts the local service). If needed:
   ```bash
   ollama serve
   ```
4. Verify:
   ```bash
   ollama run llama3.2:3b
   ```

Default model is `llama3.2:3b`. Change it with `OLLAMA_MODEL` in `.env` or as an environment variable. The Python scripts call Ollama's local HTTP API at `http://localhost:11434`.

**No Ollama needed for Task 1 token counting or the NumPy attention implementation.** Task 3's Hugging Face examples download model weights the first time and may need several GB of disk space. Set `RUN_HF=0` to skip those downloads.

## 4. Run each task

Run commands from the project root:

```bash
python task1_models_tokens_cost/main.py
python task2_capabilities_limitations/main.py
python task3_transformer_concepts/main.py
python task4_prompt_toolkit/main.py
```

Task 2 and Task 4 call the local Ollama model. Task 4 measures classification accuracy against the included 15-ticket golden set, prompt variants, sampling settings, chain-of-thought prompting and Pydantic extraction. Results are written under each task's `outputs/` folder.

## 5. Configuration

Copy `.env.example` to `.env` and edit as needed. Never commit `.env`.

- `OLLAMA_BASE_URL`: local Ollama endpoint
- `OLLAMA_MODEL`: model tag available from `ollama list`
- `RUN_HF`: set to `0` to skip Hugging Face demos

## 6. Notes about reproducibility and evaluation

- LLM outputs can vary by model version. Classification calls use temperature `0` but can still differ across model versions.
- The expected scores in the assignment are examples, not guaranteed targets. The scripts calculate and report actual accuracy.
- The recent-event cutoff test records the model's response; it is not a reliable way to establish the model's actual training cutoff.
- The arithmetic and logic answers are verified with deterministic Python checks where possible.
- Token counts depend on the tokenizer encoding. `cl100k_base` is used as a practical common baseline, not as the tokenizer for every model family.
- Prices in Task 1 are editable illustrative input values. Verify current official pricing before submitting a real estimate.
- Do not put personal resumes, real customer tickets, API keys, or confidential data into prompts.

## 7. Git

```bash
git init
git add .
git commit -m "Add Week 4 LLM fundamentals assessment"
```

Commit results and documentation separately. Confirm `.env` is ignored before pushing.
