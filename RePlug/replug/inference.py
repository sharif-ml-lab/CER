from datasets import load_from_disk
from RePlug.retriever.faiss_index import FaissRetriever
from RePlug.retriever.retriever import DenseRetriever
from RePlug.replug.llm_client import HuggingFaceClient
from RePlug.utils.tools import aggregate_token_probs, construct_prompt, prediction_is_correct, calculate_doc_scores
import asyncio
import aiohttp
import torch
from tqdm import tqdm


async def main(dataset_path, index_path, chunks_path, model_name, n=500, seed=11,
               question_col='question', choices_col='choices', answer_col='answer'):
    # Load and prepare dataset
    sampled_dataset = load_and_prepare_dataset(dataset_path, n, seed)

    # Initialize retrievers and client
    faiss_retriever, dense_retriever, client = initialize_retrievers_and_client(index_path, chunks_path, model_name)

    # Calculate accuracy
    acc = await calculate_accuracy(sampled_dataset, faiss_retriever, dense_retriever, client, question_col, choices_col,
                                   answer_col)

    # Print the final accuracy
    print("Final Accuracy:", acc / len(sampled_dataset))


def load_and_prepare_dataset(dataset_path, n, seed):
    # Load dataset
    dataset = load_from_disk(dataset_path)
    dataset = dataset['test']

    # Shuffle the dataset with a specific seed
    shuffled_dataset = dataset.shuffle(seed=seed)

    # Take the first 'n' samples
    sampled_dataset = shuffled_dataset.select(range(n))
    return sampled_dataset


def initialize_retrievers_and_client(index_path, chunks_path, model_name):
    # Initialize retrievers
    faiss_retriever = FaissRetriever(index_path, chunks_path, metric='cosine')
    dense_retriever = DenseRetriever(model_name="facebook/contriever")
    client = HuggingFaceClient(model_name=model_name)
    return faiss_retriever, dense_retriever, client


async def calculate_accuracy(sampled_dataset, faiss_retriever, dense_retriever, client, question_col, choices_col,
                             answer_col):
    acc = 0
    faulty_res = 0
    progress_bar = tqdm(total=len(sampled_dataset), desc='Calculating Accuracy', position=0, leave=True)

    # Iterate over records in the dataset
    for i, record in enumerate(sampled_dataset):
        question = record[question_col]
        choices = record[choices_col]
        answer = record[answer_col]

        # Generate embedding for the question using DenseRetriever
        query_embedding = dense_retriever.get_embedding(question)

        # Retrieve the top 10 most similar documents using FaissRetriever
        similar_docs, raw_distances = faiss_retriever.retrieve(query_embedding, top_n=10)

        scores = calculate_doc_scores(raw_distances)

        DOC_BASE = False
        if DOC_BASE:
            similar_docs = ["\n\n".join(similar_docs)]
            # similar_docs = [similar_docs[0]]

        # Construct messages for each retrieved document
        messages = [construct_prompt(question, doc, choices, client) for doc in similar_docs]

        # Get final result text from the client
        final_result_text = await generate_response(client, messages, scores)

        # Check if the prediction is correct
        if final_result_text.lower().strip() in ['a', 'b', 'c', 'd']:
            if prediction_is_correct(final_result_text, answer, 'mmlu'):
                acc += 1
        else:
            faulty_res += 1

        # Update the progress bar and display running accuracy
        progress_bar.set_postfix(running_acc=acc / (i + 1))
        progress_bar.update(1)

    progress_bar.close()
    return acc


async def generate_response(client, messages, scores, max_token=32):
    input_ids_list = [client.tokenizer(message, return_tensors="pt").input_ids for message in messages]
    result_tokens = []

    for _ in range(max_token):
        # Send async requests to the model and get log probability answers concurrently
        async with aiohttp.ClientSession() as session:
            tasks = [client.next_prob(input_ids, prob_mode=True) for input_ids in input_ids_list]
            responses = await asyncio.gather(*tasks)

        final_probs = aggregate_token_probs(probs=responses, scores=scores, mode='ensemble')

        next_token_id = client.next_token_id(final_probs)
        result_tokens.append(next_token_id.item())

        # Append the generated token to the sequence
        input_ids_list = [torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1) for input_ids in input_ids_list]

        # Stop if the model generates the end-of-sequence token
        if next_token_id.item() == client.tokenizer.eos_token_id:
            break

    return client.decode_result(result_tokens)


if __name__ == "__main__":
    import argparse

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Run document retrieval and question answering.")
    parser.add_argument('--dataset_path', default='/data/Ideas/data/datasets/mmlu.hf', type=str,
                        help="Path to the dataset.")
    parser.add_argument('--index_path', default="/data/Ideas/data/wiki_embeddings.index", type=str,
                        help="Path to the FAISS index.")
    parser.add_argument('--chunks_path', default="/data/Ideas/data/wiki_chunk_texts.pkl", type=str,
                        help="Path to the chunk texts.")
    parser.add_argument('--model_name', default="/data/TensorRT-LLM/Meta-Llama-3.1-70B-Instruct", type=str,
                        help="Path to the HuggingFace model.")
    parser.add_argument('--n', type=int, default=500, help="Number of samples to process.")
    parser.add_argument('--seed', type=int, default=11, help="Seed for shuffling the dataset.")
    parser.add_argument('--question_col', type=str, default='question', help="Column name for questions.")
    parser.add_argument('--choices_col', type=str, default='choices', help="Column name for choices.")
    parser.add_argument('--answer_col', type=str, default='answer', help="Column name for answers.")

    args = parser.parse_args()

    # Run the main function with parsed arguments
    asyncio.run(main(dataset_path=args.dataset_path,
                     index_path=args.index_path,
                     chunks_path=args.chunks_path,
                     model_name=args.model_name,
                     n=args.n,
                     seed=args.seed,
                     question_col=args.question_col,
                     choices_col=args.choices_col,
                     answer_col=args.answer_col))
