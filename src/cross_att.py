import torch
import torch.nn.functional as F


def cross_attention_refinement(
    features: torch.Tensor,
    queries: torch.Tensor,
    temp: float,
):
    """
    Cross-attention between frame features and segment/action queries.

    Args:
        features: (B, T, D) frame features
        queries:  (B, S, D) decoder outputs (segment embeddings/ action queries)
        temp:     temperature used for softmax

    Returns:
        features_refined: (B, T, D) refined frame features
        attn_weights:     (B, T, S) attention weights
    """
    D = features.size(-1)

    attn_scores = torch.bmm(features, queries.transpose(1, 2))  # (B, T, S)
    attn_scores = attn_scores / (D ** 0.5)
    attn_weights = F.softmax(attn_scores / temp, dim=-1)

    features_refined = torch.bmm(attn_weights, queries)
    features_refined = F.normalize(features + features_refined, dim=-1)

    return features_refined, attn_weights
