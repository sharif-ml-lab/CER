import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class DenseRetriever(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(DenseRetriever, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        embeddings = self.encoder(**inputs).last_hidden_state
        return torch.mean(embeddings, dim=1)  # Mean pooling over tokens

    def get_embedding(self, text):
        return self.forward([text]).detach()
