from RePlug.replug.inference import main
import asyncio

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
