# config.py

# You can define multiple configurations in a list or dictionary.
# Each config entry here is a distinct run configuration.
# Adjust as needed for your own experiments.

multi_run_configs = [
    {
        "run_name": "Self Const",
        "model_name": "/home/dev/models/Meta-Llama-3.1-8B-Instruct",
        "k": 10,
        "aggregate": True,
        "decoding_mode": 'new',  # new, baseline
        "baseline_cot": None,  # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "scoring_mode": 'log',  # log, min, max, h_mean
        "sampling_mode": "temp",  # cot, temp
        "confidence_calculation_mode": "default"  # Options: "default", "sum", "entropy", "top_2_diff"
    },
    {
        "run_name": "CoT Decoding",
        "model_name": "/home/dev/models/Meta-Llama-3.1-8B-Instruct",
        "k": 10,
        "aggregate": True,
        "decoding_mode": 'baseline',  # new, baseline
        "baseline_cot": 0,  # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "scoring_mode": 'log',  # log, min, max, h_mean
        "sampling_mode": "cot",  # cot, temp
        "confidence_calculation_mode": "top_2_diff"  # Options: "default", "sum", "entropy", "top_2_diff"
    },
    {
        "run_name": "Ours + Temp + Conf",
        "model_name": "/home/dev/models/Meta-Llama-3.1-8B-Instruct",
        "k": 10,
        "aggregate": True,
        "decoding_mode": 'new',  # new, baseline
        "baseline_cot": 0,  # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "scoring_mode": 'log',  # log, min, max, h_mean
        "sampling_mode": "temp",  # cot, temp
        "confidence_calculation_mode": "top_2_diff"  # Options: "default", "sum", "entropy", "top_2_diff"
    },
    {
        "run_name": "Ours + CoT + Conf",
        "model_name": "/home/dev/models/Meta-Llama-3.1-8B-Instruct",
        "k": 10,
        "aggregate": True,
        "decoding_mode": 'new',  # new, baseline
        "baseline_cot": 0,  # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "scoring_mode": 'log',  # log, min, max, h_mean
        "sampling_mode": "cot",  # cot, temp
        "confidence_calculation_mode": "default"  # Options: "default", "sum", "entropy", "top_2_diff"
    },
    {
        "run_name": "Ours + Temp + Conf + min",
        "model_name": "/home/dev/models/Meta-Llama-3.1-8B-Instruct",
        "k": 10,
        "aggregate": True,
        "decoding_mode": 'new',  # new, baseline
        "baseline_cot": 0,  # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "scoring_mode": 'min',  # log, min, max, h_mean
        "sampling_mode": "temp",  # cot, temp
        "confidence_calculation_mode": "top_2_diff"  # Options: "default", "sum", "entropy", "top_2_diff"
    },
    {
        "run_name": "Ours + Temp + Conf + max",
        "model_name": "/home/dev/models/Meta-Llama-3.1-8B-Instruct",
        "k": 10,
        "aggregate": True,
        "decoding_mode": 'new',  # new, baseline
        "baseline_cot": 0,  # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "scoring_mode": 'max',  # log, min, max, h_mean
        "sampling_mode": "temp",  # cot, temp
        "confidence_calculation_mode": "top_2_diff"  # Options: "default", "sum", "entropy", "top_2_diff"
    },
    {
        "run_name": "Ours + Temp + Conf + hmean",
        "model_name": "/home/dev/models/Meta-Llama-3.1-8B-Instruct",
        "k": 10,
        "aggregate": True,
        "decoding_mode": 'new',  # new, baseline
        "baseline_cot": 0,  # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "scoring_mode": 'h_mean',  # log, min, max, h_mean
        "sampling_mode": "temp",  # cot, temp
        "confidence_calculation_mode": "top_2_diff"  # Options: "default", "sum", "entropy", "top_2_diff"
    },
    {
        "run_name": "Ours + Temp + + P + log",
        "model_name": "/home/dev/models/Meta-Llama-3.1-8B-Instruct",
        "k": 10,
        "aggregate": True,
        "decoding_mode": 'new',  # new, baseline
        "baseline_cot": 0,  # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "scoring_mode": 'log',  # log, min, max, h_mean
        "sampling_mode": "temp",  # cot, temp
        "confidence_calculation_mode": "sum"  # Options: "default", "sum", "entropy", "top_2_diff"
    },
    {
        "run_name": "Ours + Temp + * P + log",
        "model_name": "/home/dev/models/Meta-Llama-3.1-8B-Instruct",
        "k": 10,
        "aggregate": True,
        "decoding_mode": 'new',  # new, baseline
        "baseline_cot": 0,  # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "scoring_mode": 'log',  # log, min, max, h_mean
        "sampling_mode": "temp",  # cot, temp
        "confidence_calculation_mode": "default"  # Options: "default", "sum", "entropy", "top_2_diff"
    },
    {
        "run_name": "Ours + Temp + Weighted Conf + log",
        "model_name": "/home/dev/models/Meta-Llama-3.1-8B-Instruct",
        "k": 10,
        "aggregate": True,
        "decoding_mode": 'new',  # new, baseline
        "baseline_cot": 0,  # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "scoring_mode": 'log',  # log, min, max, h_mean
        "sampling_mode": "temp",  # cot, temp
        "confidence_calculation_mode": "top_2_diff_weighted"
        # Options: "default", "sum", "entropy", "top_2_diff", top_2_diff_weighted
    },
    {
        "run_name": "Ours + Temp + H + log",
        "model_name": "/home/dev/models/Meta-Llama-3.1-8B-Instruct",
        "k": 10,
        "aggregate": True,
        "decoding_mode": 'new',  # new, baseline
        "baseline_cot": 0,  # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "scoring_mode": 'log',  # log, min, max, h_mean
        "sampling_mode": "temp",  # cot, temp
        "confidence_calculation_mode": "entropy"
        # Options: "default", "sum", "entropy", "top_2_diff", top_2_diff_weighted
    },
    {
        "run_name": "Ours + Temp + +P + min",
        "model_name": "/home/dev/models/Meta-Llama-3.1-8B-Instruct",
        "k": 10,
        "aggregate": True,
        "decoding_mode": 'new',  # new, baseline
        "baseline_cot": 0,  # 0 for regular CoT, 1 for greedy number COT , None for self-consistency
        "scoring_mode": 'min',  # log, min, max, h_mean
        "sampling_mode": "temp",  # cot, temp
        "confidence_calculation_mode": "sum"
        # Options: "default", "sum", "entropy", "top_2_diff", top_2_diff_weighted
    }

]

# You can include more config entries here if desired.
