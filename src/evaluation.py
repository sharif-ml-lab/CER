from tqdm import tqdm

from src.self_consistency import self_consistency_decode
from src.decoding import cot_decode
from src.greedy_on_numbers import greedy_number_cot_decode
from src.utils import load_model_and_tokenizer, construct_prompt, print_final_accuracy, save_results_to_csv, \
    load_and_sample_parquet_datasets
from src.config import Config, multi_run_configs


# evaluate the model on a single example
def evaluate_single_example(model, tokenizer, question, correct_answer_str, k, aggregate, decoding_mode, scoring_mode,
                            baseline_cot, sampling_mode, few_shot, few_shot_path, confidence_method):
    messages = [{"role": "user",
                 "content": construct_prompt(question=question, few_shot=few_shot, few_shot_path=few_shot_path)}]

    # pick k-branch and continue each path with greedy sampling.
    if baseline_cot == "k-branch" or baseline_cot == "k-seperate":
        result, confidence, final_ans = cot_decode(
            model,
            tokenizer,
            messages,
            aggregate_paths=aggregate,
            k=k,
            decoding_mode=decoding_mode,
            sampling_mode=sampling_mode,
            scoring_mode=scoring_mode,
            baseline_cot=baseline_cot,
            confidence_method=confidence_method,
        )

    # elif baseline_cot == "greedy_decoding": # ??????????????????
    #     result, confidence, final_ans = greedy_number_cot_decode( # greedy
    #         model,
    #         tokenizer,
    #         messages,
    #         aggregate_paths=aggregate,
    #         k=k,
    #         sampling_mode=sampling_mode,
    #         scoring_mode=scoring_mode
    #     )

    elif baseline_cot == "self_consistency":
        result, confidence, final_ans = self_consistency_decode(model, tokenizer, messages, k=k)

    try:
        model_answer = float(final_ans)
        correct_answer = float(correct_answer_str)
        is_correct = ((model_answer - correct_answer) <= 1e-2)
    except ValueError:
        is_correct = False

    return {
        'question': question,
        'correct_answer': correct_answer_str,
        'predicted_answer': result,
        'predicted_final_answer': final_ans,
        'confidence_score': confidence,
        'is_correct': is_correct
    }


# evaluate the model on dataset
def evaluate_dataset(model, tokenizer, dataset, k, aggregate, decoding_mode, description, scoring_mode, baseline_cot,
                     sampling_mode, few_shot, few_shot_path, confidence_method):
    total_questions = len(dataset)
    correct_answers = 0
    results = []

    with tqdm(total=total_questions, desc=f"Processing {description}", dynamic_ncols=True) as pbar:
        for idx, example in dataset.iterrows():
            question = example['question']
            correct_answer = str(example['numeric_final_answer'])

            result_dict = evaluate_single_example(model, tokenizer, question, correct_answer, k, aggregate,
                                                  decoding_mode, scoring_mode, baseline_cot, sampling_mode, few_shot,
                                                  few_shot_path, confidence_method)
            results.append(result_dict)

            if result_dict['is_correct']:
                correct_answers += 1

            running_accuracy = (correct_answers / (idx + 1)) * 100
            pbar.set_postfix(idx=idx + 1, running_accuracy=f"{running_accuracy:.2f}%")
            pbar.update(1)

    save_results_to_csv(results, f"{description}_evaluation_results.csv")
    accuracy = (correct_answers / total_questions) * 100
    print_final_accuracy(description, accuracy)
    return accuracy


def run_dataset(config: Config):
    model_name = config.model_name
    aggregate = config.aggregate
    K = config.K
    few_shot = config.few_shot
    number_samples = config.number_samples
    seed = config.seed
    read_model_from_local = config.read_model_from_local
    data_dir = config.data_dir
    run_name = config.run_name

    model, tokenizer = load_model_and_tokenizer(model_name, read_model_from_local)

    dataset_files = config.datasets

    loaded_datasets = load_and_sample_parquet_datasets(data_dir, dataset_files, number_samples=number_samples,
                                                       seed=seed)

    # Loop over each config
    for cfg_run_name, cfg in multi_run_configs.items():
        if run_name == cfg_run_name or run_name == "all":
            print("======================================")
            print(f"Running: {cfg_run_name}")
            print(f"Cofing: {cfg}")
            print("======================================")

            # Evaluate on each of the loaded datasets
            for dataset_name, dataset_df in loaded_datasets.items():
                print(f"\nEvaluating {dataset_name} using {cfg_run_name} ...")

                if few_shot:
                    if dataset_name == "allenai":
                        few_shot_path = config.allenai_shtos
                    elif dataset_name == "open_math":
                        few_shot_path = config.open_math_shtos
                    elif dataset_name == "multiarith":
                        few_shot_path = config.multiarith_shots
                    elif dataset_name == "metamath":
                        few_shot_path = config.metamath_shots
                    elif dataset_name == "gsm8k":
                        few_shot_path = config.gsm8k_shots

                evaluate_dataset(
                    model,
                    tokenizer,
                    dataset_df,
                    k=K,
                    aggregate=aggregate,
                    decoding_mode=cfg['decoding_mode'],
                    description=f"{dataset_name}_{cfg_run_name}",
                    scoring_mode=cfg['scoring_mode'],
                    baseline_cot=cfg['baseline_cot'],
                    sampling_mode=cfg['sampling_mode'],
                    confidence_method=cfg['confidence'],
                    few_shot=few_shot,
                    few_shot_path=few_shot_path,
                )

            print(f"Finished run: {cfg_run_name}")
            print("======================================\n")
