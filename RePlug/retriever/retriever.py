import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import os


class DenseRetriever(nn.Module):
    def __init__(self, model_name="facebook/contriever"):
        super(DenseRetriever, self).__init__()

        cache_dir = os.path.expanduser('/home/dev/.cache/huggingface/hub/')

        self.encoder = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True).to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)

    def forward(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to('cuda')
        embeddings = self.encoder(**inputs).last_hidden_state
        return torch.mean(embeddings, dim=1)  # Mean pooling over tokens

    def get_embedding(self, text):
        return self.forward([text]).detach().cpu().numpy()
