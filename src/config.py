from dataclasses import dataclass

# list of different settings to run
multi_run_configs = {
    "Self Const": {
        "decoding_mode": 'all',  # "all": all the numbers "last": the last number 
        "baseline_cot": 'self_consistency',  # [k-branch, k-seperate, self_consistency]
        "scoring_mode": 'log',  # log, min, max, h_mean
        "sampling_mode": "temperature",  # "temperature": temperature sampling  "greedy": greedy sampling
        "confidence": "default"  # Options: "default", "sum", "entropy", "top_2_diff"
    },

    "CoT Decoding": {
        "decoding_mode": 'last',  # "all": all the numbers "last": the last number 
        "baseline_cot": "k-branch", # [k-branch, k-seperate, self_consistency]
        "scoring_mode": 'log',  # log, min, max, h_mean
        "sampling_mode": "greedy",  # "temperature": temperature sampling  "greedy": greedy sampling # (I'm not sure which one is correct?)
        "confidence": "top_2_diff"  # Options: "default", "sum", "entropy", "top_2_diff"
    },

    "Ours + Temp + Conf": {
        "decoding_mode": 'all',  # "all": all the numbers "last": the last number 
        "baseline_cot": "k-seperate",  # [k-branch, k-seperate, self_consistency]
        "scoring_mode": 'log',  # log, min, max, h_mean
        "sampling_mode": "temperature",  # "temperature": temperature sampling  "greedy": greedy sampling
        "confidence": "top_2_diff"  # Options: "default", "sum", "entropy", "top_2_diff"
    },

    "Ours + CoT + Conf": {
        "decoding_mode": 'all',  # "all": all the numbers "last": the last number 
        "baseline_cot": "k-branch",  # [k-branch, k-seperate, self_consistency]
        "scoring_mode": 'log',  # log, min, max, h_mean
        "sampling_mode": "temperature", # "temperature": temperature sampling  "greedy": greedy sampling
        "confidence": "default"  # Options: "default", "sum", "entropy", "top_2_diff"
    },

    "Ours + Temp + Conf + min": {
        "decoding_mode": 'all',  # "all": all the numbers "last": the last number 
        "baseline_cot": "k-seperate",  # [k-branch, k-seperate, self_consistency]
        "scoring_mode": 'min',  # log, min, max, h_mean
        "sampling_mode": "temperature",  # "temperature": temperature sampling  "greedy": greedy sampling
        "confidence": "top_2_diff"  # Options: "default", "sum", "entropy", "top_2_diff"
    },

    "Ours + Temp + Conf + max": {
        "decoding_mode": 'all',  # "all": all the numbers "last": the last number 
        "baseline_cot": "k-seperate",  # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "scoring_mode": 'max',  # log, min, max, h_mean
        "sampling_mode": "temperature",  # "temperature": temperature sampling  "greedy": greedy sampling
        "confidence": "top_2_diff"  # Options: "default", "sum", "entropy", "top_2_diff"
    },

    "Ours + Temp + Conf + hmean": {
        "decoding_mode": 'all',  # "all": all the numbers "last": the last number 
        "baseline_cot": "k-seperate",  # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "scoring_mode": 'h_mean',  # log, min, max, h_mean
        "sampling_mode": "temperature",  # "temperature": temperature sampling  "greedy": greedy sampling
        "confidence": "top_2_diff"  # Options: "default", "sum", "entropy", "top_2_diff"
    },

    "Ours + Temp + + P + log": {
        "decoding_mode": 'all',  # "all": all the numbers "last": the last number 
        "baseline_cot": "k-seperate",  # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "scoring_mode": 'log',  # log, min, max, h_mean
        "sampling_mode": "temperature",  # "temperature": temperature sampling  "greedy": greedy sampling
        "confidence": "sum"  # Options: "default", "sum", "entropy", "top_2_diff"
    },

    "Ours + Temp + * P + log": {
        "decoding_mode": 'all',  # "all": all the numbers "last": the last number 
        "baseline_cot": "k-seperate",  # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "scoring_mode": 'log',  # log, min, max, h_mean
        "sampling_mode": "temperature",  # "temperature": temperature sampling  "greedy": greedy sampling
        "confidence": "default"  # Options: "default", "sum", "entropy", "top_2_diff"
    },

    "Ours + Temp + Weighted Conf + log": {
        "decoding_mode": 'all',  # "all": all the numbers "last": the last number 
        "baseline_cot": "k-seperate",  # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "scoring_mode": 'log',  # log, min, max, h_mean
        "sampling_mode": "temperature",  # "temperature": temperature sampling  "greedy": greedy sampling
        "confidence": "top_2_diff_weighted" # Options: "default", "sum", "entropy", "top_2_diff", top_2_diff_weighted
    },

    "Ours + Temp + H + log": {
        "decoding_mode": 'all',  # "all": all the numbers "last": the last number 
        "baseline_cot": "k-seperate",  # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "scoring_mode": 'log',  # log, min, max, h_mean
        "sampling_mode": "temperature",  # "temperature": temperature sampling  "greedy": greedy sampling
        "confidence": "entropy" # Options: "default", "sum", "entropy", "top_2_diff", top_2_diff_weighted
    },

    "Ours + Temp + +P + min": {
        "decoding_mode": 'all',  # "all": all the numbers "last": the last number 
        "baseline_cot": "k-seperate",  # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "scoring_mode": 'min',  # log, min, max, h_mean
        "sampling_mode": "temperature",   # "temperature": temperature sampling  "greedy": greedy sampling
        "confidence": "sum" # Options: "default", "sum", "entropy", "top_2_diff", top_2_diff_weighted
    }
}


# general configuration
@dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"  # Path to the HuggingFace model or local directory
    data_dir = "data"
    read_model_from_local: bool = False # Load the model from the local directory instead of the HF.
    hugging_face_token: str = "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" # Huggingface Token

    run_name = "CoT Decoding" # specify the running mode "all" that means all of them.
    K: int = 10 # number of chains in self-consistency or number of branching in cot-decoding
    aggregate: bool = True # True: aggregate paths False: the best path

    few_shot: bool = True # True: few-shot False: zero-shot
    number_samples: int = 500  # Number of samples to process
    seed: int = 11  # Seed for shuffling the dataset

    gsm8k_shots: str = "inputs/shots/gsm8k.txt" # path to shots of gsm8k
    multiarith_shots: str = "inputs/shots/multiarith.txt" # path to shots of multiarith
    allenai_shtos: str ="inputs/shots/allenai.txt" # path to shots of allenai
    open_math_shtos: str ="inputs/shots/open_math.txt" # path to shots of open_math
    metamath_shots: str ="inputs/shots/metamath.txt" # path to shots of metamath




