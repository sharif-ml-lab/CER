import faiss
import numpy as np


def build_faiss_index(doc_embeddings):
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(doc_embeddings)
    return index


def update_index(index, new_embeddings):
    index.add(new_embeddings)
