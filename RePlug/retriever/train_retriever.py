import torch
import torch.optim as optim
import torch.nn.functional as F
from retriever.retriever import DenseRetriever
from utils.data_loader import load_training_data
from replug.lm_api import get_lm_likelihood


def train_retriever(epochs):
    retriever = DenseRetriever()
    optimizer = optim.Adam(retriever.parameters(), lr=2e-5)
    training_data = load_training_data()  # Load training data containing context-document pairs

    for epoch in range(epochs):
        for context, true_continuation, corpus in training_data:
            # Step 1: Retrieve documents
            doc_embeddings = [retriever.get_embedding(doc) for doc in corpus]
            context_embedding = retriever.get_embedding(context)
            similarities = [torch.cosine_similarity(context_embedding, doc_emb, dim=0) for doc_emb in doc_embeddings]

            # Step 2: Compute Retrieval Likelihood and LM Likelihood
            lm_scores = [get_lm_likelihood(context + " " + doc, true_continuation) for doc in corpus]

            # Normalize scores using softmax for KL divergence
            pr = F.softmax(torch.tensor(similarities), dim=0)
            ql = F.softmax(torch.tensor(lm_scores), dim=0)

            # Step 3: Compute Loss
            loss = F.kl_div(pr.log(), ql, reduction='batchmean')

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
