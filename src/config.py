from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv(override=True)  # Reads .env file and loads environment variables

# list of different settings to run
multi_run_configs = {
    # "Branch Greedy Special Case": {
    #     "decoding_mode": 'all',  # "all": all the numbers "last": the last number
    #     # seperated_greedy_special, branch_greedy_special
    #     "baseline_cot": 'branch_greedy_special',
    #     "scoring_mode": 'log',  # log, min, max, h_mean, mean, weighted_mean
    #     # "temperature": temperature sampling  "greedy": greedy sampling
    #     "sampling_mode": "temperature",
    #     "confidence": "default"  # Options: "default", "sum", "entropy", "top_2_diff"
    # },

    # "Seperated Greedy Special Case": {
    #     "decoding_mode": 'all',  # "all": all the numbers "last": the last number
    #     # seperated_greedy_special, branch_greedy_special
    #     "baseline_cot": 'seperated_greedy_special',
    #     "scoring_mode": 'log',  # log, min, max, h_mean, mean, weighted_mean
    #     # "temperature": temperature sampling  "greedy": greedy sampling
    #     "sampling_mode": "temperature",
    #     "confidence": "default"  # Options: "default", "sum", "entropy", "top_2_diff"
    # },

    # "Self Const": {
    #     "decoding_mode": 'all',  # "all": all the numbers "last": the last number
    #     # [k-branch, k-seperate, self_consistency, p_true]
    #     "baseline_cot": 'self_consistency',
    #     "scoring_mode": 'log',  # log, min, max, h_mean, mean, weighted_mean
    #     # "temperature": temperature sampling  "greedy": greedy sampling
    #     "sampling_mode": "temperature",
    #     "confidence": "default",  # Options: "default", "sum", "entropy", "top_2_diff",
    #     "use_base_prompt": True,
    # },


    # "P_True": {
    #     "decoding_mode": 'all',  # "all": all the numbers "last": the last number
    #     # [k-branch, k-seperate, self_consistency, p_true]
    #     "baseline_cot": "p_true",
    #     "scoring_mode": 'log',  # log, min, max, h_mean, mean, weighted_mean
    #     # "temperature": temperature sampling  "greedy": greedy sampling
    #     "sampling_mode": "temperature",
    #     "confidence": "top_2_diff",  # Options: "default", "sum", "entropy", "top_2_diff"
    #     "use_base_prompt": True,
    # },

    # "CoT Decoding": {
    #     "decoding_mode": 'last',  # "all": all the numbers "last": the last number
    #     # [k-branch, k-seperate, self_consistency, p_true]
    #     "baseline_cot": "k-branch",
    #     "scoring_mode": 'log',  # log, min, max, h_mean, mean, weighted_mean
    #     "sampling_mode": "greedy",
    #     # "temperature": temperature sampling  "greedy": greedy sampling # (I'm not sure which one is correct?)
    #     "confidence": "top_2_diff",  # Options: "default", "sum", "entropy", "top_2_diff"
    #     "use_base_prompt": True,
    # },

    "Ours + Temp + Conf": {
        "decoding_mode": 'all',  # "all": all the numbers "last": the last number
        # [k-branch, k-seperate, self_consistency, p_true]
        "baseline_cot": "k-seperate",
        "scoring_mode": 'log',  # log, min, max, h_mean, mean, weighted_mean
        # "temperature": temperature sampling  "greedy": greedy sampling
        "sampling_mode": "temperature",
        "confidence": "top_2_diff"  # Options: "default", "sum", "entropy", "top_2_diff"
    },

    # "Ours + CoT + Conf": {
    #     "decoding_mode": 'all',  # "all": all the numbers "last": the last number
    #     "baseline_cot": "k-branch",  # [k-branch, k-seperate, self_consistency, p_true]
    #     "scoring_mode": 'log',  # log, min, max, h_mean, mean, weighted_mean
    #     # "temperature": temperature sampling  "greedy": greedy sampling
    #     "sampling_mode": "temperature",
    #     "confidence": "default"  # Options: "default", "sum", "entropy", "top_2_diff"
    # },

    "Ours + Temp + Conf + min": {
        "decoding_mode": 'all',  # "all": all the numbers "last": the last number
        # [k-branch, k-seperate, self_consistency, p_true]
        "baseline_cot": "k-seperate",
        "scoring_mode": 'min',  # log, min, max, h_mean, mean, weighted_mean
        # "temperature": temperature sampling  "greedy": greedy sampling
        "sampling_mode": "temperature",
        "confidence": "top_2_diff"  # Options: "default", "sum", "entropy", "top_2_diff"
    },

    # "Ours + Temp + Conf + max": {
    #     "decoding_mode": 'all',  # "all": all the numbers "last": the last number
    #     # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
    #     "baseline_cot": "k-seperate",
    #     "scoring_mode": 'max',  # log, min, max, h_mean, mean, weighted_mean
    #     # "temperature": temperature sampling  "greedy": greedy sampling
    #     "sampling_mode": "temperature",
    #     "confidence": "top_2_diff"  # Options: "default", "sum", "entropy", "top_2_diff"
    # },

    "Ours + Temp + Conf + hmean": {
        "decoding_mode": 'all',  # "all": all the numbers "last": the last number
        # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "baseline_cot": "k-seperate",
        "scoring_mode": 'h_mean',  # log, min, max, h_mean, mean, weighted_mean
        # "temperature": temperature sampling  "greedy": greedy sampling
        "sampling_mode": "temperature",
        "confidence": "top_2_diff"  # Options: "default", "sum", "entropy", "top_2_diff"
    },

    # "Ours + Temp + + P + log": {
    #     "decoding_mode": 'all',  # "all": all the numbers "last": the last number
    #     # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
    #     "baseline_cot": "k-seperate",
    #     "scoring_mode": 'log',  # log, min, max, h_mean, mean, weighted_mean
    #     # "temperature": temperature sampling  "greedy": greedy sampling
    #     "sampling_mode": "temperature",
    #     "confidence": "sum"  # Options: "default", "sum", "entropy", "top_2_diff"
    # },

    "Ours + Temp + * P + log": {
        "decoding_mode": 'all',  # "all": all the numbers "last": the last number
        # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "baseline_cot": "k-seperate",
        "scoring_mode": 'log',  # log, min, max, h_mean, mean, weighted_mean
        # "temperature": temperature sampling  "greedy": greedy sampling
        "sampling_mode": "temperature",
        "confidence": "default"  # Options: "default", "sum", "entropy", "top_2_diff"
    },

    "Ours + Temp + Weighted Conf + log": {
        "decoding_mode": 'all',  # "all": all the numbers "last": the last number
        # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "baseline_cot": "k-seperate",
        "scoring_mode": 'log',  # log, min, max, h_mean, mean, weighted_mean
        # "temperature": temperature sampling  "greedy": greedy sampling
        "sampling_mode": "temperature",
        # Options: "default", "sum", "entropy", "top_2_diff", top_2_diff_weighted
        "confidence": "top_2_diff_weighted"
    },

    # "Ours + Temp + H + log": {
    #     "decoding_mode": 'all',  # "all": all the numbers "last": the last number
    #     # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
    #     "baseline_cot": "k-seperate",
    #     "scoring_mode": 'log',  # log, min, max, h_mean, mean, weighted_mean
    #     # "temperature": temperature sampling  "greedy": greedy sampling
    #     "sampling_mode": "temperature",
    #     # Options: "default", "sum", "entropy", "top_2_diff", top_2_diff_weighted
    #     "confidence": "entropy"
    # },

    # "Ours + Temp + +P + min": {
    #     "decoding_mode": 'all',  # "all": all the numbers "last": the last number
    #     # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
    #     "baseline_cot": "k-seperate",
    #     "scoring_mode": 'min',  # log, min, max, h_mean, mean, weighted_mean
    #     # "temperature": temperature sampling  "greedy": greedy sampling
    #     "sampling_mode": "temperature",
    #     # Options: "default", "sum", "entropy", "top_2_diff", top_2_diff_weighted
    #     "confidence": "sum"
    # },

    "Ours + Temp + Conf + Weighted_Mean": {
        "decoding_mode": 'all',  # "all": all the numbers "last": the last number
        # [k-branch, k-seperate, self_consistency, p_true]
        "baseline_cot": "k-seperate",
        "scoring_mode": 'weighted_mean',  # log, min, max, h_mean, mean, weighted_mean
        # "temperature": temperature sampling  "greedy": greedy sampling
        "sampling_mode": "temperature",
        "confidence": "top_2_diff"  # Options: "default", "sum", "entropy", "top_2_diff"
    },

    "Ours + Temp + *P + min": {
        "decoding_mode": 'all',  # "all": all the numbers "last": the last number
        # [k-branch, k-seperate, self_consistency, p_true]
        "baseline_cot": "k-seperate",
        "scoring_mode": 'min',  # log, min, max, h_mean, mean, weighted_mean
        # "temperature": temperature sampling  "greedy": greedy sampling
        "sampling_mode": "temperature",
        "confidence": "default"  # Options: "default", "sum", "entropy", "top_2_diff"
    },

    "Ours + Temp + *P + hmean": {
        "decoding_mode": 'all',  # "all": all the numbers "last": the last number
        # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "baseline_cot": "k-seperate",
        "scoring_mode": 'h_mean',  # log, min, max, h_mean, mean, weighted_mean
        # "temperature": temperature sampling  "greedy": greedy sampling
        "sampling_mode": "temperature",
        "confidence": "default"  # Options: "default", "sum", "entropy", "top_2_diff"
    },


    ############ test ##########
    # "Ours + Temp + *P + hmean + extension": {
    #     "decoding_mode": 'all',  # "all": all the numbers "last": the last number
    #     # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
    #     "baseline_cot": "k-seperate",
    #     "scoring_mode": 'h_mean',  # log, min, max, h_mean, mean, weighted_mean
    #     # "temperature": temperature sampling  "greedy": greedy sampling
    #     "sampling_mode": "temperature",
    #     # Options: "default", "sum", "entropy", "top_2_diff"
    #     "confidence": "top_2_diff_extension",
    # },
}


# general configuration
@dataclass
class Config:
    model_name: str = os.getenv("MODEL_NAME",
                                "meta-llama/Llama-3.1-8B-Instruct")  # Path to the HuggingFace model or local directory
    data_dir = os.getenv("DATA_DIR", "data")
    # Load the model from the local directory instead of the HF.
    read_model_from_huggingface: bool = eval(
        os.getenv("LOCAL_MODEL", 'True'))
    hugging_face_token: str = os.getenv(
        "HUGGING_FACE_TOKEN", "")  # Huggingface Token

    # specify the running mode "all" that means all of them.
    run_name: str = os.getenv("RUN_NAME", "Ours + Temp + Conf")
    # number of chains in self-consistency or number of branching in cot-decoding
    K: int = int(os.getenv("K", 10))
    aggregate: bool = True  # True: aggregate paths False: the best path
    multihop: bool = eval(os.getenv("MULTIHOP", 'False'))

    # whether to use random selection setting or not.
    random_selection: bool = eval(os.getenv("RANDOM", "False"))
    # number of words to be selected randomly.
    random_selection_number_words: int = int(os.getenv("RANDOM_NUM", 5))

    # True: few-shot False: zero-shot
    few_shot: bool = eval(os.getenv("FEW_SHOT", 'True'))
    # Number of samples to process
    number_samples: int = int(os.getenv("N_SAMPLE", 500))
    seed: int = int(os.getenv("SEED", 11))  # Seed for shuffling the dataset
    step_decomposition: bool = eval(os.getenv("STEP_DECOMPOSITION", 'False'))

    gsm8k_shots: str = "inputs/shots/gsm8k.txt"  # path to shots of gsm8k
    multiarith_shots: str = "inputs/shots/multiarith.txt"  # path to shots of multiarith
    allenai_shots: str = "inputs/shots/allenai.txt"  # path to shots of allenai
    math_shots: str = "inputs/shots/math.txt"  # path to shots of math
    hotpot_shots: str = "inputs/shots/hotpot.txt"  # path to shots of hotpot
    trivia_shots: str = "inputs/shots/trivia.txt"  # path to shots of trivia

    datasets = eval(os.getenv("DATASETS", """{
        "allenai": "allenai_math_qa_test_processed.parquet",
        "multiarith": "ChilleD_MultiArith_combine_processed.parquet",
        "math": "src_datasets_math_dataset_test_processed.parquet",
        "gsm8k": "openai_gsm8k_test_processed.parquet",
        "hotpot": "hotpotqa_processed.parquet",
        "trivia": "triviaqa_processed.parquet",
        "popqa": "popqa_processed.parquet",
    }"""))

    batch_size = int(os.getenv("BATCH_SIZE", 1))
