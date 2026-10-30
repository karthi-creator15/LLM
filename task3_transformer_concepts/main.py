import os
import numpy as np
import matplotlib.pyplot as plt

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def single_head_attention(x, seed=7):
    rng = np.random.default_rng(seed)
    d_model = x.shape[1]
    Wq, Wk, Wv = (rng.normal(size=(d_model, d_model)) for _ in range(3))
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    scores = Q @ K.T / np.sqrt(d_model)
    weights = softmax(scores, axis=1)
    return weights, weights @ V

def multi_head_attention(x, n_heads=2, seed=11):
    rng = np.random.default_rng(seed)
    seq_len, d_model = x.shape
    assert d_model % n_heads == 0
    head_dim = d_model // n_heads
    Wq = rng.normal(size=(n_heads, d_model, head_dim))
    Wk = rng.normal(size=(n_heads, d_model, head_dim))
    Wv = rng.normal(size=(n_heads, d_model, head_dim))
    outputs, weights_all = [], []
    for h in range(n_heads):
        q, k, v = x @ Wq[h], x @ Wk[h], x @ Wv[h]
        weights = softmax((q @ k.T) / np.sqrt(head_dim), axis=1)
        outputs.append(weights @ v)
        weights_all.append(weights)
    concatenated = np.concatenate(outputs, axis=1)
    Wo = rng.normal(size=(n_heads * head_dim, d_model))
    return weights_all, concatenated @ Wo

def positional_encoding(max_positions=10, d_model=8):
    positions = np.arange(max_positions)[:, None]
    dims = np.arange(d_model)[None, :]
    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    pe = np.zeros((max_positions, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe

def main():
    x = np.array([[1., 0., 1., 0.],
                  [0., 2., 0., 2.],
                  [1., 1., 1., 1.],
                  [2., 0., 2., 0.]])
    weights, output = single_head_attention(x)
    print("Attention weights (4x4):")
    print(np.round(weights, 4))
    print("Row sums:", weights.sum(axis=1))
    assert np.allclose(weights.sum(axis=1), 1.0)

    heads, multi_output = multi_head_attention(x)
    print("\nTwo head attention matrices:")
    for i, head in enumerate(heads, start=1):
        print(f"Head {i}:\n{np.round(head, 4)}")
        assert np.allclose(head.sum(axis=1), 1.0)
    print("Concatenated + output projection shape:", multi_output.shape)

    pe = positional_encoding()
    plt.figure(figsize=(9, 5))
    plt.imshow(pe, aspect="auto")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Position")
    plt.title("Sinusoidal positional encoding (10 positions × 8 dimensions)")
    plt.colorbar(label="Encoding value")
    plt.tight_layout()
    plt.savefig("task3_transformer_concepts/positional_encoding.png", dpi=150)
    print("\nSaved positional_encoding.png")

    if os.getenv("RUN_HF", "1") == "1":
        try:
            from transformers import pipeline
            fill_mask = pipeline("fill-mask", model="google-bert/bert-base-uncased")
            print("\nBERT fill-mask example:")
            print(fill_mask("The capital of France is [MASK].")[:3])
            generator = pipeline("text-generation", model="openai-community/gpt2")
            print("\nGPT-2 text-generation example:")
            print(generator("A transformer model can", max_new_tokens=35,
                             do_sample=False, pad_token_id=50256)[0]["generated_text"])
        except Exception as exc:
            print("Hugging Face demo skipped/failed:", exc)
            print("Check internet access, torch/transformers installation and available RAM.")
    else:
        print("RUN_HF=0: Hugging Face model downloads skipped.")

    print("\nENCODER VS DECODER – six lines")
    notes = [
        "Encoder-only models such as BERT use bidirectional context to build representations of input text.",
        "They are commonly used for classification, named-entity recognition and fill-mask tasks.",
        "Decoder-only models such as GPT use causal attention and predict the next token.",
        "They are commonly used for text generation, chat and code completion.",
        "Transformers allow parallel processing of tokens during training, unlike sequential RNN recurrence.",
        "Self-attention models long-range relationships more directly and scales effectively with data and compute."
    ]
    print("\n".join(notes))

if __name__ == "__main__":
    main()
