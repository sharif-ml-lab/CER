import re
from datasets import load_dataset, DatasetDict, concatenate_datasets
import pandas as pd


# Function to extract the last numeric value from a string
def extract_last_numeric_value(text):
    matches = re.findall(r'\b\d+\.?\d*\b', text)
    return matches[-1] if matches else None


def preprocess_math_qa(df, answer_column):
    options_column, correct_column = answer_column

    # Function to get the numeric final answer from the options and correct columns
    def get_numeric_final_answer(row):
        options = row[options_column]
        correct = row[correct_column].strip().lower()

        # Parse the options
        options_dict = {}
        for option in options.split(','):
            key, value = option.strip().split(')')
            key = key.strip().lower()
            value = value.strip()
            options_dict[key] = value

        # Get the correct answer
        final_answer = options_dict.get(correct, None)
        if final_answer:
            # Replace comma with dot for float values and check if the answer is numeric
            final_answer = final_answer.replace(',', '.')
            return extract_last_numeric_value(final_answer)
        return None

    # Apply the function to each row to get the numeric final answer
    df['numeric_final_answer'] = df.apply(get_numeric_final_answer, axis=1)

    return df


def preprocess_meta_math_qa(df, answer_column):
    def extractor(text):
        first, ans = text.split("The answer is:")
        if ans.replace(',', '.').isnumeric():
            return ans
        else:
            return None

    return df[answer_column].apply(extractor)


def preprocess_mmlu(df, answer_column):
    choices_column, answer_column = answer_column

    # Function to get the numeric final answer from the choices and answer columns
    def get_numeric_final_answer(row):
        choices = row[choices_column]
        answer_info = row[answer_column].split()

        if len(answer_info) == 2:
            try:
                answer_index = int(answer_info.strip()[0])
                if 0 <= answer_index < len(choices):
                    final_answer = choices[answer_index].replace(',', '.')
                    return extract_last_numeric_value(final_answer)
            except ValueError:
                return None
        return None

    # Apply the function to each row to get the numeric final answer
    df['numeric_final_answer'] = df.apply(get_numeric_final_answer, axis=1)

    return df


def preprocess_open_math_instruct(df, answer_column):
    def extractor(text):
        if text.isnumeric():
            return text
        else:
            return None

    return df[answer_column].apply(extractor)


def preprocess_gsm8k(df, answer_column):
    def extractor(text):
        first, ans = text.split('####')
        if ans.isnumeric():
            return ans
        else:
            return None

    return df[answer_column].apply(extractor)


def preprocess_multi_arith(df, answer_column):
    def extractor(text):
        if text.isnumeric():
            return text
        else:
            return None

    return df[answer_column].apply(extractor)


# Function to process and save a dataset
def process_and_save_dataset(dataset_info, save_path):
    dataset_name = dataset_info["dataset_name"]
    answer_column = dataset_info["answer_column"]
    preprocess_function = dataset_info["preprocess_function"]

    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Combine all splits into one
    combined_dataset = concatenate_datasets([ds for split, ds in dataset.items()])

    # Convert to pandas DataFrame for easier manipulation
    df = combined_dataset.to_pandas()

    # Apply the specific preprocess function to find the numeric final answer
    df['numeric_final_answer'] = preprocess_function(df, answer_column)

    # Filter out rows without a numeric final answer
    df = df.dropna(subset=['numeric_final_answer'])

    # Save the processed dataset in Parquet format to save disk space
    df.to_parquet(f"{save_path}/{dataset_name.replace('/', '_')}_processed.parquet", index=False)


if __name__ == "__main__":
    # Dictionary mapping dataset names to their specific preprocess functions and answer columns
    datasets_to_process = [
        {"dataset_name": "allenai/math_qa", "answer_column": "options, correct",
         "preprocess_function": preprocess_math_qa},
        {"dataset_name": "meta-math/MetaMathQA", "response": "solution",
         "preprocess_function": preprocess_meta_math_qa},
        {"dataset_name": "cais/mmlu", "answer_column": "choices, answer", "preprocess_function": preprocess_mmlu},
        {"dataset_name": "nvidia/OpenMathInstruct-2", "expected_answer": "result",
         "preprocess_function": preprocess_open_math_instruct},
        {"dataset_name": "openai/gsm8k", "answer_column": "answer", "preprocess_function": preprocess_gsm8k},
        {"dataset_name": "ChilleD/MultiArith", "answer_column": "final_ans",
         "preprocess_function": preprocess_multi_arith}
    ]

    # Path to save processed datasets
    save_path = "./data"

    # Process each dataset
    for dataset_info in datasets_to_process:
        process_and_save_dataset(dataset_info, save_path)
