from datasets import load_from_disk
from RePlug.retriever.faiss_index import FaissRetriever
from RePlug.retriever.retriever import DenseRetriever
from RePlug.replug.llm_client import HuggingFaceClient
from RePlug.utils.tools import aggregate_token_probs
import asyncio
import aiohttp
import torch


async def main():
    dataset = load_from_disk('/var/csi/rawfile/Ideas/data/datasets/nq.hf')
    dataset = dataset['validation']

    # Initialize retrievers
    index_path = "/var/csi/rawfile/Ideas/data/wiki_embeddings.index"  # Path to the FAISS index
    chunks_path = "/var/csi/rawfile/Ideas/data/wiki_chunk_texts.pkl"  # Path to the chunk texts
    faiss_retriever = FaissRetriever(index_path, chunks_path, metric='cosine')

    dense_retriever = DenseRetriever(model_name="facebook/contriever")

    client = HuggingFaceClient(model_name="/home/dev/llama3.1/Meta-Llama-3.1-8B-Instruct")

    # Iterate over records in the dataset
    for i, record in enumerate(dataset):
        question = record['question']['text']

        # Generate embedding for the question using DenseRetriever
        query_embedding = dense_retriever.get_embedding(question)

        # Retrieve the top 10 most similar documents using FaissRetriever
        similar_docs, scores = faiss_retriever.retrieve(query_embedding, top_n=10)

        # Construct messages for each retrieved document
        messages = [f"Knowledge: {doc}\nQuestion: {question}" for doc in similar_docs]

        input_ids_list = [client.tokenizer(message, return_tensors="pt").input_ids for message in
                          messages]
        max_token = 128
        result_tokens = []

        for _ in range(max_token):
            # Send 10 async requests to OpenAI GPT-4 model and get log probability answers concurrently
            async with aiohttp.ClientSession() as session:
                tasks = [client.next_prob(input_ids, prob_mode=True) for input_ids in input_ids_list]
                responses = await asyncio.gather(*tasks)

            final_probs = aggregate_token_probs(probs=responses, scores=scores, mode='ensemble')

            next_token_id = client.next_token_id(final_probs)
            result_tokens.append(next_token_id.item())

            # Append the generated token to the sequence
            input_ids_list = [torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1) for input_ids in
                              input_ids_list]

            # Stop if the model generates the end-of-sequence token
            if next_token_id.item() == client.tokenizer.eos_token_id:
                break

        final_result_text = client.decode_result(result_tokens)

        print(question)
        print("ANS")
        print(final_result_text)

        exit(0)


if __name__ == "__main__":
    asyncio.run(main())
