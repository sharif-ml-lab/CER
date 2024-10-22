from retriever.retriever import DenseRetriever
from retriever.faiss_index import build_faiss_index
from replug.input_reformulation import reformulate_inputs
from replug.lm_api import query_lm
import numpy as np


def replug_inference(context, corpus, retriever_model):
    retriever = retriever_model
    embeddings = [retriever.get_embedding(doc).numpy() for doc in corpus]
    index = build_faiss_index(np.array(embeddings))

    # Retrieve top-k documents
    context_embedding = retriever.get_embedding(context).numpy()
    _, retrieved_indices = index.search(np.array([context_embedding]), k=5)
    retrieved_docs = [corpus[i] for i in retrieved_indices[0]]

    # Input reformulation and LM query
    predictions = reformulate_inputs(context, retrieved_docs)
    return predictions
