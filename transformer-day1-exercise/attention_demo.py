import numpy as np
from typing import Tuple


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
) -> np.ndarray:
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)

    Returns:
        Attention output of shape (seq_len, d_v)
    """
    d_k = Q.shape[-1]
    raw_scores = Q @ K.T
    scaled_scores = raw_scores / np.sqrt(d_k)
    attention_weights = softmax(scaled_scores)
    return attention_weights @ V


if __name__ == "__main__":
    np.random.seed(42)
    seq_len = 6
    d_k = 8
    d_v = 8

    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_v)

    output = scaled_dot_product_attention(Q, K, V)

    print(f"Input shapes  — Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
    print(f"Output shape  — {output.shape}")
    print(f"Output:\n{np.round(output, 4)}")
