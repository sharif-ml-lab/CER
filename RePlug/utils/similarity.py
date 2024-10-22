import torch
import torch.nn.functional as F

def cosine_similarity(embedding1, embedding2):
    """
    Compute the cosine similarity between two embeddings.
    """
    return F.cosine_similarity(embedding1, embedding2, dim=0).item()
