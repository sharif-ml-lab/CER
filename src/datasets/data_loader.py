from datasets import load_dataset, concatenate_datasets
import os
from dotenv import load_dotenv

load_dotenv()  # Reads .env file and loads environment variables


# Function to check if a string is a valid number (integer or float)
def is_valid_number(s):
    try:
        float(s.strip())
        return True
    except ValueError:
        return False


def preprocess_math(df, answer_column, old_question_column, new_question_column):
    """
    Preprocess function for the MATH dataset to extract numeric answers.

    Args:
        df (pd.DataFrame): Input dataframe containing the dataset.
        answer_column (str): Column name containing the solution.
        old_question_column (str): Column name of the original question text.
        new_question_column (str): New column name for the question.

    Returns:
        pd.DataFrame: Preprocessed dataframe with numeric answers.
    """

    def get_numeric_final_answer(row):
        """
        Extract numeric final answer from the solution text.

        Args:
            row (pd.Series): A row of the DataFrame.

        Returns:
            str or None: Extracted numeric answer, or None if not found.
        """
        solution = row[answer_column]

        # Extract the numeric value using keywords and parsing
        try:
            # Look for boxed answer indicated by '\\boxed{' or similar markers
            if '\\boxed{' in solution:
                start_idx = solution.index('\\boxed{') + len('\\boxed{')
                end_idx = solution.index('}', start_idx)
                answer = solution[start_idx:end_idx]
            else:
                # Fallback: Try to locate the final numeric value heuristically
                answer = solution.split()[-1].strip()

            # Remove potential formatting issues and validate as numeric
            answer = answer.replace(',', '.').strip()
            if is_valid_number(answer):
                return answer
        except Exception as e:
            print(f"Error extracting numeric final answer: {e}, row: {row}")
        return None

    # Apply the numeric answer extraction
    df['numeric_final_answer'] = df.apply(get_numeric_final_answer, axis=1)

    # Rename the question column
    df = df.rename(columns={old_question_column: new_question_column})

    return df


# Preprocess functions for each dataset
def preprocess_math_qa(df, answer_column, old_question_column, new_question_column):
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
    df = df.rename(columns={old_question_column: new_question_column})
    return df


def preprocess_gsm8k(df, answer_column, old_question_column, new_question_column):
    answer_column = answer_column

    def get_numeric_final_answer(row):
        answer = row[answer_column]
        _, answer = answer.split("####")
        answer = answer.replace(',', '.')
        if is_valid_number(answer):
            return answer
        return None

    df['numeric_final_answer'] = df.apply(get_numeric_final_answer, axis=1)
    df = df.rename(columns={old_question_column: new_question_column})
    return df


def preprocess_multi_arith(df, answer_column, old_question_column, new_question_column):
    final_answer_column = answer_column

    def get_numeric_final_answer(row):
        final_answer = row[final_answer_column]
        final_answer = final_answer.replace(',', '.')
        if is_valid_number(final_answer):
            return final_answer
        return None

    df['numeric_final_answer'] = df.apply(get_numeric_final_answer, axis=1)
    df = df.rename(columns={old_question_column: new_question_column})
    return df


# Function to process and save a dataset
def process_and_save_dataset(dataset_info, save_path):
    dataset_name = dataset_info["dataset_name"]
    split = dataset_info["split"]
    answer_column = dataset_info["answer_column"]
    old_question_column = dataset_info["old_question_column"]
    new_question_column = dataset_info["new_question_column"]
    preprocess_function = dataset_info["preprocess_function"]
    config_name = dataset_info.get("config_name", None)

    # Load the dataset split
    if config_name:
        dataset = load_dataset(
            dataset_name, config_name, split=split, trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)

    # Convert to pandas DataFrame for easier manipulation
    df = dataset.to_pandas()

    # Apply the specific preprocess function to find the numeric final answer
    df = preprocess_function(
        df, answer_column, old_question_column, new_question_column)

    # Filter out rows without a numeric final answer
    df = df.dropna(subset=['numeric_final_answer'])

    # Print the number of records before saving
    print(
        f"Number of records in {dataset_name} ({split} split) after preprocessing: {len(df)}")

    # Save the processed dataset in Parquet format to save disk space
    df.to_parquet(
        f"{save_path}/{dataset_name.replace('/', '_').replace('.py', '')}_{split}_processed.parquet", index=False)


if __name__ == '__main__':
    # Dictionary mapping dataset names to their specific preprocess functions and answer columns
    datasets_to_process = [
        {"dataset_name": "allenai/math_qa", "split": "test", "answer_column": ("options", "correct"),
         "preprocess_function": preprocess_math_qa,
         "old_question_column": "Problem",
         "new_question_column": "question",
         },
        {"dataset_name": "openai/gsm8k", "split": "test", "answer_column": "answer", "config_name": "main",
         "preprocess_function": preprocess_gsm8k,
         "old_question_column": "question",
         "new_question_column": "question",
         },
        {"dataset_name": "ChilleD/MultiArith", "split": "test", "answer_column": "final_ans",
         "preprocess_function": preprocess_multi_arith,
         "old_question_column": "question",
         "new_question_column": "question"
         },
        {"dataset_name": "src/datasets/math_dataset.py", "split": "test", "answer_column": "solution",
         "preprocess_function": preprocess_math,
         "old_question_column": "problem",
         "new_question_column": "question"
         },
    ]

    # Path to save processed datasets
    save_path = os.getenv("DATA_DIR", "data")

    # Process each dataset
    for dataset_info in datasets_to_process:
        process_and_save_dataset(dataset_info, save_path)
