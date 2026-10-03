import torch
import torch.nn.functional as F

def sgns_loss(center_vec: torch.Tensor, pos_vec: torch.Tensor,
              neg_vecs: torch.Tensor) -> torch.Tensor:
    """
    Returns the scalar float64 SGNS loss.
    """
    positive_score = torch.dot(center_vec, pos_vec)
    negative_scores = neg_vecs @ center_vec
    return F.softplus(-positive_score) + F.softplus(negative_scores).sum()