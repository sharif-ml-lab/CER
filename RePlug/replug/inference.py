from datasets import load_from_disk
from RePlug.retriever.faiss_index import FaissRetriever
from RePlug.retriever.retriever import DenseRetriever
from RePlug.replug.llm_client import OpenAIClient
from RePlug.utils.evaluation import calculate_result
import asyncio
import aiohttp


async def main():
    dataset = load_from_disk('/home/aquasar/Desktop/Ideas/data/datasets/nq.hf')
    dataset = dataset['validation']

    # Initialize retrievers
    index_path = "/home/aquasar/Desktop/Ideas/data/wiki_embeddings.index"  # Path to the FAISS index
    chunks_path = "/home/aquasar/Desktop/Ideas/data/wiki_chunks.pkl"  # Path to the chunk texts
    faiss_retriever = FaissRetriever(index_path, chunks_path, metric='cosine')

    dense_retriever = DenseRetriever(model_name="facebook/contriever")

    # Initialize OpenAI client
    api_key = "YOUR_OPENAI_API_KEY"
    gpt4_client = OpenAIClient(model_name="gpt-4o", api_key=api_key)

    # Iterate over records in the dataset
    for i, record in enumerate(dataset):
        question = record['question']

        # Generate embedding for the question using DenseRetriever
        query_embedding = dense_retriever.get_embedding(question)

        # Retrieve the top 10 most similar documents using FaissRetriever
        similar_docs = faiss_retriever.retrieve(query_embedding, top_n=10)

        # Construct messages for each retrieved document
        messages = [f"Knowledge: {doc}\nQuestion: {question}" for doc in similar_docs]

        # Send 10 async requests to OpenAI GPT-4 model and get log probability answers concurrently
        async with aiohttp.ClientSession() as session:
            tasks = [gpt4_client.call(message, prob_mode=True) for message in messages]
            responses = await asyncio.gather(*tasks)

        # Pass responses to black box function 'calculate_score'
        # (This function is assumed to be available and implemented elsewhere)
        final_score = calculate_result(responses)
        print(f"Final Score for Question {i + 1}: {final_score}")

        # Limiting to 5 iterations for the sake of example
        if i >= 4:
            break


if __name__ == "__main__":
    asyncio.run(main())
