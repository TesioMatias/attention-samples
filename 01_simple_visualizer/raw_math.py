# /// script
# dependencies = [
#   "numpy",
# ]
# ///

import numpy as np

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def interactive_attention_visualizer():
    print("--- Project 1: Interactive Attention Calculator ---")
    user_input = input("Enter a short sentence (e.g., 'The cat sat'): ")
    
    # 1. Tokenization: Splitting the sentence into a list of words
    words = user_input.split()
    if not words:
        print("Please enter at least one word.")
        return

    # 2. Embedding Simulation: 
    # In a real model, words are mapped to high-dimensional vectors.
    # Here, we create a 4D vector for each word using random values for visualization.
    np.random.seed(42)
    d_k = 4
    embeddings = np.random.randn(len(words), d_k)
    
    print(f"\nCaptured {len(words)} tokens. Simulated {d_k}-dimensional embeddings created.")

    # 3. Q, K, V Assignments
    # We use the embeddings for all three, simulating a basic self-attention block.
    Q = embeddings
    K = embeddings
    V = embeddings

    # 4. Scaled Dot-Product Calculation
    # We compute raw scores using Q multiplied by the transpose of K: Q K^T.
    raw_scores = np.matmul(Q, K.T)
    
    # Scale by the square root of d_k to maintain training stability.
    scaled_scores = raw_scores / np.sqrt(d_k)

    # 5. Attention Weights (Softmax)
    # The scores are converted into probabilities that sum to 1.
    attention_weights = softmax(scaled_scores)

    print("\n--- ATTENTION WEIGHTS (Softmax) ---")
    print("This matrix shows how much each word (row) focuses on every other word (column).")
    
    # Header for the matrix
    header = " " * 12 + "  ".join([f"{w:>10}" for w in words])
    print(header)
    
    for i, row in enumerate(attention_weights):
        row_str = "  ".join([f"{val:10.4f}" for val in row])
        print(f"{words[i]:>10}: [{row_str}]")

    # 6. Final Output (Context Vectors)
    # The output is the weighted sum of the Values: $Attention(Q, K, V) = softmax(\frac{Q K^T}{\sqrt{d_k}}) V$.
    output = np.matmul(attention_weights, V)
    
    print("\n--- CONTEXT-ENRICHED OUTPUT ---")
    print("New vector representations that now 'know' their neighbors:")
    for i, vec in enumerate(output):
        print(f"{words[i]:>10}: {vec}")

if __name__ == "__main__":
    interactive_attention_visualizer()