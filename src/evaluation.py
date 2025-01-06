from tqdm import tqdm

from src.self_consistency import self_consistency_decode
from src.decoding import cot_decode
from src.greedy_on_numbers import greedy_number_cot_decode
from src.utils import load_model_and_tokenizer, construct_prompt, print_final_accuracy, save_results_to_csv, \
    load_and_sample_parquet_datasets
from src.config import Config, multi_run_configs


# evaluate the model on a batch of exapmles
def evaluate_batch_examples(
        model,
        tokenizer,
        batch_questions,
        batch_correct_answers,
        k,
        aggregate,
        decoding_mode,
        scoring_mode,
        baseline_cot,
        sampling_mode,
        few_shot,
        few_shot_path,
        confidence_method,
        multihop
):
    # Construct a list of messages for each question in the batch
    batch_messages = []
    for question in batch_questions:
        batch_messages.append([{"role": "user", "content": construct_prompt(
            question=question,
            few_shot=few_shot,
            few_shot_path=few_shot_path)}])

    # Depending on baseline_cot, call the appropriate batch decoding function
    if baseline_cot in ("k-branch", "k-seperate"):
        # These functions return lists of results, confidences, and final answers
        batch_results = cot_decode(
            model,
            tokenizer,
            batch_messages,
            aggregate_paths=aggregate,
            k=k,
            decoding_mode=decoding_mode,
            sampling_mode=sampling_mode,
            scoring_mode=scoring_mode,
            baseline_cot=baseline_cot,
            confidence_method=confidence_method,
            multihop=multihop
        )
    elif baseline_cot == "self_consistency":
        batch_results = self_consistency_decode(
            model,
            tokenizer,
            batch_messages,
            k=k
        )
    else:
        raise ValueError(f"Unsupported baseline_cot mode: {baseline_cot}")

    # Build the output list with evaluation details
    batch_output = []
    for i, question in enumerate(batch_questions):
        predicted_text, confidence_score, predicted_final_answer = batch_results[i]
        correct_answer_str = batch_correct_answers[i]

        # Compare numeric answers if both are valid floats
        try:
            if not multihop:
                model_answer = float(predicted_final_answer)
                correct_answer = float(correct_answer_str)
                is_correct = abs(model_answer - correct_answer) <= 1e-2
            else:
                model_answer = predicted_final_answer
                correct_answer = correct_answer_str
                is_correct = model_answer.lower() == correct_answer.lower()
        except ValueError:
            is_correct = False

        batch_output.append({
            "question": question,
            "correct_answer": correct_answer_str,
            "predicted_answer": predicted_text,
            "predicted_final_answer": predicted_final_answer,
            "confidence_score": confidence_score,
            "is_correct": is_correct
        })

    return batch_output


# evaluate the model on dataset
def evaluate_dataset(
        model,
        tokenizer,
        dataset,
        k,
        aggregate,
        decoding_mode,
        description,
        scoring_mode,
        baseline_cot,
        sampling_mode,
        few_shot,
        few_shot_path,
        confidence_method,
        batch_size,
        multihop,
):
    # Extract lists of questions and answers directly from the dataframe
    questions = dataset["question"].tolist()

    if not multihop:
        correct_answers_list = dataset["numeric_final_answer"].astype(
            str).tolist()
    else:
        correct_answers_list = dataset["answer"].astype(str).tolist()

    total_questions = len(questions)
    correct_answers = 0
    results = []

    # Process the dataset in batches
    with tqdm(total=total_questions, desc=f"Processing {description}", dynamic_ncols=True) as pbar:
        for start_idx in range(0, total_questions, batch_size):
            end_idx = min(start_idx + batch_size, total_questions)

            # Slice out the batch
            batch_questions = questions[start_idx:end_idx]
            batch_correct_answers = correct_answers_list[start_idx:end_idx]

            print(batch_questions)
            print(batch_correct_answers)

            # Evaluate the batch
            batch_results = evaluate_batch_examples(
                model,
                tokenizer,
                batch_questions,
                batch_correct_answers,
                k,
                aggregate,
                decoding_mode,
                scoring_mode,
                baseline_cot,
                sampling_mode,
                few_shot,
                few_shot_path,
                confidence_method,
                multihop
            )

            # Accumulate results and update correct answers count
            for result_dict in batch_results:
                results.append(result_dict)
                if result_dict["is_correct"]:
                    correct_answers += 1

            running_accuracy = (correct_answers / int(end_idx)) * 100
            pbar.set_postfix(idx=int(end_idx),
                             running_accuracy=f"{running_accuracy:.2f}%")

            pbar.update(end_idx - start_idx)

    # Save and print final results
    save_results_to_csv(
        results, f"outputs/{description}_evaluation_results_{'few_shot' if few_shot else 'zero_shot'}.csv")
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
    read_model_from_huggingface = config.read_model_from_huggingface
    data_dir = config.data_dir
    run_name = config.run_name
    batch_size = config.batch_size
    dataset_files = config.datasets
    multihop = config.multihop

    model, tokenizer = load_model_and_tokenizer(
        model_name, read_model_from_huggingface)

    loaded_datasets = load_and_sample_parquet_datasets(data_dir, dataset_files, number_samples=number_samples,
                                                       seed=seed)

    # Loop over each config
    for cfg_run_name, cfg in multi_run_configs.items():
        if run_name == cfg_run_name or run_name == "all":
            print("======================================")
            print(f"Running: {cfg_run_name}")
            print(f"Confing: {cfg}")
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
                    elif dataset_name == "hotpot":
                        few_shot_path = config.hotpot_shots
                    elif dataset_name == "trivia":
                        few_shot_path = config.trivia_shots
                    else:
                        raise ValueError(
                            'You have to provide the examples for the prompt')
                else:
                    # it should be run in a zero-shot format
                    few_shot_path = None

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
                    batch_size=batch_size,
                    multihop=multihop,
                )

            print(f"Finished run: {cfg_run_name}")
            print("======================================\n")
