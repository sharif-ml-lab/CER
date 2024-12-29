import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from Decoding import cot_decode
from greedy_on_numbers import greedy_number_cot_decode
from self_consistency import self_consistency_decode
from tqdm import tqdm
from typing import List
from config import multi_run_configs  # Import the list of multiple configs


def load_model_and_tokenizer(model_name: str):
    """
    Load the model and tokenizer from the specified path.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
        device_map='cuda',
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def construct_prompt(question: str) -> str:
    """
    Construct a prompt using a given question.
    """
    base = f"""Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

Q: {question}
A: """
    return base


def evaluate_single_example(
        model,
        tokenizer,
        question: str,
        correct_answer_str: str,
        k: int,
        aggregate: bool,
        decoding_mode: str,
        scoring_mode: str,
        COT: int,
        sampling_mode: str,
        confidence_mode: str
) -> dict:
    """
    Evaluate the model on a single example.
    """
    messages = [{"role": "user", "content": construct_prompt(question)}]

    if COT == 0:
        result, confidence, final_ans = cot_decode(
            model,
            tokenizer,
            messages,
            sampling_mode=sampling_mode,
            scoring_mode=scoring_mode,
            k=k,
            decoding_mode=decoding_mode,
            num_beams=1,
            max_new_tokens=512,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.0,
            length_penalty=1.0,
            no_repeat_ngram_size=0,
            early_stopping=False,
            aggregate_paths=aggregate,
            confidence_mode=confidence_mode
        )
    elif COT == 1:
        result, confidence, final_ans = greedy_number_cot_decode(
            model,
            tokenizer,
            messages,
            aggregate_paths=aggregate,
            max_new_tokens=512,
            k=k,
            sampling_mode=sampling_mode,
            scoring_mode=scoring_mode,
        )
    else:
        result, confidence, final_ans = self_consistency_decode(
            model,
            tokenizer,
            messages,
            aggregate_paths=aggregate,
            k=k,
            decoding_mode=decoding_mode
        )

    # Check correctness
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


def evaluate_dataset(
        model,
        tokenizer,
        dataset: pd.DataFrame,
        k: int,
        aggregate: bool,
        decoding_mode: str,
        description: str,
        scoring_mode: str,
        COT: int,
        sampling_mode: str,
        confidence_mode: str
) -> float:
    """
    Evaluate the model on the given dataset.
    """
    total_questions = len(dataset)
    correct_answers = 0
    results = []

    with tqdm(total=total_questions, desc=f"Processing {description}", dynamic_ncols=True) as pbar:
        for idx, example in dataset.iterrows():
            question = example['question']
            correct_answer = str(example['numeric_final_answer'])  # ensure it's a string for parsing downstream

            result_dict = evaluate_single_example(
                model, tokenizer, question, correct_answer,
                k, aggregate, decoding_mode, scoring_mode, COT, sampling_mode, confidence_mode
            )
            results.append(result_dict)

            if result_dict['is_correct']:
                correct_answers += 1

            running_accuracy = (correct_answers / (int(idx) + 1)) * 100
            pbar.set_postfix(idx=int(idx) + 1, running_accuracy=f"{running_accuracy:.2f}%")
            pbar.update(1)

    save_results_to_csv(results, f"{description}_evaluation_results_{sampling_mode}_{decoding_mode}.csv")
    accuracy = (correct_answers / total_questions) * 100
    print_final_accuracy(description, accuracy)
    return accuracy


def save_results_to_csv(results: List[dict], filename: str):
    """
    Save evaluation results to a CSV file.
    """
    results_df = pd.DataFrame(results)
    results_df.to_csv(filename, index=False)


def print_final_accuracy(description: str, accuracy: float):
    """
    Print the final accuracy of the evaluation.
    """
    print(f"Final Accuracy for {description}: {accuracy:.2f}%")


def load_and_sample_parquet_datasets(data_dir: str, dataset_files: list, n: int, seed: int):
    """
    Load multiple Parquet datasets from the given list of filenames, shuffle each dataset
    based on the given seed, and select up to n records. If a dataset contains fewer than
    n records, select all.
    """
    loaded_datasets = {}
    for filename in dataset_files:
        full_path = os.path.join(data_dir, filename)
        if os.path.isfile(full_path):
            df = pd.read_parquet(full_path)
            # Shuffle with the given seed
            df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
            # Select up to n records
            if len(df) > n:
                df = df.head(n)
            # Keep the loaded DataFrame in a dictionary
            loaded_datasets[filename] = df
        else:
            print(f"File not found: {full_path}")
    return loaded_datasets


if __name__ == '__main__':
    # Specify the directory with your Parquet files
    data_dir = "/home/dev/Ideas/data"

    # List of dataset filenames
    dataset_files = [
        "allenai_math_qa_processed.parquet",
        # "nvidia_OpenMathInstruct-2_processed.parquet",
        # "ChilleD_MultiArith_processed.parquet",
        # "meta-math_MetaMathQA_processed.parquet",
        # "openai_gsm8k_processed.parquet"
    ]

    # Load and sample each dataset
    loaded_datasets = load_and_sample_parquet_datasets(data_dir, dataset_files, n=500, seed=42)

    # Loop over each config
    for cfg in multi_run_configs:
        print("======================================")
        print(f"Running: {cfg['run_name']}")
        print("======================================")

        # Load the model and tokenizer for each config
        model, tokenizer = load_model_and_tokenizer(cfg["model_name"])

        # Print basic info
        print(f"Model name: {cfg['model_name']}")
        print(f"Sampling mode: {cfg['sampling_mode']}")
        print(f"Decoding mode: {cfg['decoding_mode']}")

        # Evaluate on each of the loaded datasets
        for dataset_name, dataset_df in loaded_datasets.items():
            print(f"\nEvaluating {dataset_name} using {cfg['run_name']} ...")
            evaluate_dataset(
                model,
                tokenizer,
                dataset_df,
                k=cfg['k'],
                aggregate=cfg['aggregate'],
                decoding_mode=cfg['decoding_mode'],
                description=f"{dataset_name}_{cfg['run_name']}",
                scoring_mode=cfg['scoring_mode'],
                COT=cfg['baseline_cot'],
                sampling_mode=cfg['sampling_mode'],
                confidence_mode=cfg['confidence_calculation_mode']
            )

        print(f"Finished run: {cfg['run_name']}")
        print("======================================\n")
