import re
from datasets import load_dataset, DatasetDict, concatenate_datasets
import pandas as pd


# Function to check if a string is a valid number (integer or float)
def is_valid_number(s):
    try:
        float(s.strip())
        return True
    except ValueError:
        return False


# Preprocess functions for each dataset
def preprocess_math_qa(df, answer_column):
    options_column, correct_column = answer_column

    def get_numeric_final_answer(row):
        options = row[options_column]
        correct = row[correct_column].strip().lower()

        options_dict = {}
        for option in options.split(','):
            try:
                key, value = option.strip().split(')')
                key = key.strip().lower()
                value = value.strip()
                options_dict[key] = value
            except ValueError:
                print(f"Skipping option due to split error: {option}")
                continue

        final_answer = options_dict.get(correct, None)
        if final_answer:
            final_answer = final_answer.replace(',', '.')
            if is_valid_number(final_answer):
                return final_answer
        return None

    df['numeric_final_answer'] = df.apply(get_numeric_final_answer, axis=1)
    return df


def preprocess_meta_math_qa(df, answer_column):
    def extractor(row):
        text = row[answer_column]
        try:
            first, ans = text.split("The answer is:")
            ans = ans.replace(',', '.').strip()
            if is_valid_number(ans):
                return ans
        except ValueError:
            print(f"Skipping row due to split error: {text}")
        return None

    df['numeric_final_answer'] = df.apply(extractor, axis=1)
    return df


def preprocess_mmlu(df, answer_column):
    choices_column, answer_column = answer_column

    def get_numeric_final_answer(row):
        choices = row[choices_column]
        answer_info = row[answer_column].split()

        if len(answer_info) == 2:
            try:
                answer_index = int(answer_info[0])
                if 0 <= answer_index < len(choices):
                    final_answer = choices[answer_index].replace(',', '.')
                    return is_valid_number(final_answer)
            except ValueError:
                return None
        return None

    df['numeric_final_answer'] = df.apply(get_numeric_final_answer, axis=1)
    return df


def preprocess_open_math_instruct(df, answer_column):
    result_column = answer_column

    def get_numeric_final_answer(row):
        result = row[result_column]
        result = result.replace(',', '.')
        return is_valid_number(result)

    df['numeric_final_answer'] = df.apply(get_numeric_final_answer, axis=1)
    return df


def preprocess_gsm8k(df, answer_column):
    answer_column = answer_column

    def get_numeric_final_answer(row):
        answer = row[answer_column]
        _, answer = answer.split("####")
        answer = answer.replace(',', '.')
        return is_valid_number(answer)

    df['numeric_final_answer'] = df.apply(get_numeric_final_answer, axis=1)
    return df


def preprocess_multi_arith(df, answer_column):
    final_answer_column = answer_column

    def get_numeric_final_answer(row):
        final_answer = row[final_answer_column]
        final_answer = final_answer.replace(',', '.')
        return is_valid_number(final_answer)

    df['numeric_final_answer'] = df.apply(get_numeric_final_answer, axis=1)
    return df


# Function to process and save a dataset
def process_and_save_dataset(dataset_info, save_path):
    dataset_name = dataset_info["dataset_name"]
    answer_column = dataset_info["answer_column"]
    preprocess_function = dataset_info["preprocess_function"]

    # Load the dataset
    dataset = load_dataset(dataset_name, trust_remote_code=True)

    # Combine all splits into one
    combined_dataset = concatenate_datasets([ds for split, ds in dataset.items()])

    # Convert to pandas DataFrame for easier manipulation
    df = combined_dataset.to_pandas()

    # Apply the specific preprocess function to find the numeric final answer
    df = preprocess_function(df, answer_column)

    # Filter out rows without a numeric final answer
    df = df.dropna(subset=['numeric_final_answer'])

    # Print the number of records before saving
    print(f"Number of records in {dataset_name} after preprocessing: {len(df)}")

    # Save the processed dataset in Parquet format to save disk space
    df.to_parquet(f"{save_path}/{dataset_name.replace('/', '_')}_processed.parquet", index=False)


if __name__ == '__main__':
    # Dictionary mapping dataset names to their specific preprocess functions and answer columns
    datasets_to_process = [
        {"dataset_name": "allenai/math_qa", "answer_column": ("options", "correct"),
         "preprocess_function": preprocess_math_qa},
        {"dataset_name": "meta-math/MetaMathQA", "answer_column": "response",
         "preprocess_function": preprocess_meta_math_qa},
        {"dataset_name": "cais/mmlu", "answer_column": ("choices", "answer"), "preprocess_function": preprocess_mmlu},
        {"dataset_name": "nvidia/OpenMathInstruct-2", "answer_column": "result",
         "preprocess_function": preprocess_open_math_instruct},
        {"dataset_name": "openai/gsm8k", "answer_column": "answer", "preprocess_function": preprocess_gsm8k},
        {"dataset_name": "ChilleD/MultiArith", "answer_column": "final_answer",
         "preprocess_function": preprocess_multi_arith}
    ]

    # Path to save processed datasets
    save_path = "./data"

    # Process each dataset
    for dataset_info in datasets_to_process:
        process_and_save_dataset(dataset_info, save_path)