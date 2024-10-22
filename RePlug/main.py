from retriever.train_retriever import train_retriever
from replug.inference import replug_inference
from retriever.retriever import DenseRetriever

if __name__ == "__main__":
    retriever_model = DenseRetriever()

    # Train retriever
    train_retriever(100)

    # Inference
    context = "Your input query here"
    corpus = ["Document 1", "Document 2", ...]
    response = replug_inference(context, corpus, retriever_model)
    print(response)
