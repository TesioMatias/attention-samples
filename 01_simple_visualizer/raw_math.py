import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def simple_attention_visualizer():
    # 1. Setup Input: "The cat sat"
    # We'll represent each word as a simple 4-dimensional vector (embeddings)
    words = "the cat sat"
    embeddings = np.array([
        [1.0, 0.0, 1.0, 0.0], # The
        [0.0, 1.0, 0.0, 1.0], # cat
        [1.0, 1.0, 0.0, 0.0]  # sat 
        ])
    
    d_k = embeddings.shape[1] # Dimension of keys (4)
    
    print("--- STEP 1: Input Embeddings ---")
    print(embeddings, "\n")

    # 2. Create Q, K, V (In a real model, these are learned weights)
    # For this visualizer, we'll just use the embeddings themselves
    Q = embeddings
    K = embeddings
    V = embeddings

    # 3. Calculate Raw Attention Scores (Dot Product)
    # This tells us how much each word "relates" to every other word
    # Mathematical Representation: Q * K^T
    raw_scores = np.matmul(Q, K.T)
    print("--- STEP 2: Raw Scores (Q * K^T) ---")
    print(raw_scores, "\n")

    # 4. Apply Scaling 
    # Why? To prevent vanishing gradients as discussed in Section 2.3
    scaled_scores = raw_scores / np.sqrt(d_k)
    print(f"--- STEP 3: Scaled Scores (Divided by sqrt({d_k})) ---")
    print(scaled_scores, "\n")

    # 5. Apply Softmax
    # Converts scores into probabilities (Attention Weights) that sum to 1
    attention_weights = softmax(scaled_scores)
    print("--- STEP 4: Attention Weights (Softmax) ---")
    for i, row in enumerate(attention_weights):
        formatted_row = [f"{val:.2f}" for val in row]
        print(f"{words[i]}: {formatted_row}")
    print("\n")

    # 6. Final Output (Context Vector)
    # Multiply weights by Values to get the new, context-aware representation
    output = np.matmul(attention_weights, V)
    print("--- STEP 5: Final Context-Enriched Output ---")
    print(output)

if __name__ == "__main__":
    simple_attention_visualizer()