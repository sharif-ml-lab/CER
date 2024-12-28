from dataclasses import dataclass

@dataclass
class Config:
    hugging_face_token: str = "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" # Huggingface Token
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"  # Path to the HuggingFace model
    dataset_path: str = "openai/gsm8k"  # Path to the dataset  ["MultiArith", "GSM8K"]


    baseline_cot = 1 # 0: COT; 1: GREEDY NUMBER COT; None: SELF-CONSISTENCY
    decoding_mode : str = "new"

    
    index_path: str = "/data/Ideas/data/wiki_embeddings.index"  # Path to the FAISS index
    chunks_path: str = "/data/Ideas/data/wiki_chunk_texts.pkl"  # Path to the chunk texts
    n: int = 500  # Number of samples to process
    seed: int = 11  # Seed for shuffling the dataset
    question_col: str = "question"  # Column name for questions in the dataset
    choices_col: str = "choices"  # Column name for choices in the dataset
    answer_col: str = "answer"  # Column name for answers in the dataset
