# config.py

# You can define multiple configurations in a list or dictionary.
# Each config entry here is a distinct run configuration.
# Adjust as needed for your own experiments.

multi_run_configs = [
    {
        "run_name": "config_run_1",
        "model_name": "/home/dev/models/Meta-Llama-3.1-8B-Instruct",
        "k": 10,
        "aggregate": True,
        "decoding_mode": 'new', # new, baseline
        "baseline_cot": 1,  # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "scoring_mode": 'log', # log, min, max, h_mean
        "sampling_mode": "temp", # cot, temp
        "confidence_calculation_mode": "default"  # Options: "default", "sum", "entropy", "top_2_diff"
    },
]

# You can include more config entries here if desired.