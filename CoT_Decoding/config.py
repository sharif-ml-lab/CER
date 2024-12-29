# config.py

# You can define multiple configurations in a list or dictionary.
# Each config entry here is a distinct run configuration.
# Adjust as needed for your own experiments.

multi_run_configs = [
    {
        "run_name": "config_run_1",
        "model_name": "/home/dev/models/Meta-Llama-3.1-8B-Instruct",
        "k": 5,
        "aggregate": True,
        "decoding_mode": 'new',
        "baseline_cot": 1,  # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "scoring_mode": 'log',
        "sampling_mode": "temp",
        "max_new_tokens": 256,
        "num_beams": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "early_stopping": False,
        "confidence_calculation_mode": "default"  # Options: "default", "sum", "entropy", "top_2_diff"
    },
    {
        "run_name": "config_run_2",
        "model_name": "/home/dev/models/Meta-Llama-3.1-8B-Instruct",
        "k": 10,
        "aggregate": False,
        "decoding_mode": 'baseline',
        "baseline_cot": 0,  # 0 for regular CoT
        "scoring_mode": 'log',
        "sampling_mode": "cot",
        "max_new_tokens": 512,
        "num_beams": 2,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 2,
        "early_stopping": True,
        "confidence_calculation_mode": "default"
    }
]

# You can include more config entries here if desired.