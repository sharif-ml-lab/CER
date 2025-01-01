from dataclasses import dataclass

@dataclass
class Config:
    hugging_face_token: str = "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" # Huggingface Token
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"  # Path to the HuggingFace model
    dataset_desc: str = "GSM8K"  # Path to the dataset  ["MultiArith", "GSM8K"]


    baseline_cot = "k-branch" # [k-branch, k-seperate, self_consistency]
    decoding_mode : str = "all" # "all": all the numbers "last": the last number 
    scoring_mode: str = 'log' # "log": log(1 + p) "min": min "max": max "h_mean": harmonic mean "mean": average
    aggregate: bool = True # True: aggregate paths False: the best path
    K: int = 10 # number of chains in self-consistency or number of branching in cot-decoding
    sampling_mode: str = "greedy" # "temperature": temperature sampling  "greedy": greedy sampling
    confidence: str = "cot-decoding" # "difference": p(x1) - p(x2) "selective": p(x_current) "cot-decoding": cot-decoding paper  "entropy": -sum p_i * log(p_i)

    few_shot: bool = True # True: few-shot False: zero-shot
    gsm8k_shots: str = "inputs/shots/gsm8k.txt" # path to shots of gsm8k
    multiarith_shots: str = "inputs/shots/gsm8k.txt" # path to shots of multiarith

    
    # index_path: str = "/data/Ideas/data/wiki_embeddings.index"  # Path to the FAISS index
    # chunks_path: str = "/data/Ideas/data/wiki_chunk_texts.pkl"  # Path to the chunk texts
    # n: int = 500  # Number of samples to process
    # seed: int = 11  # Seed for shuffling the dataset
    # question_col: str = "question"  # Column name for questions in the dataset
    # choices_col: str = "choices"  # Column name for choices in the dataset
    # answer_col: str = "answer"  # Column name for answers in the dataset
