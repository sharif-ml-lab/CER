from datasets import load_dataset, concatenate_datasets

# Function to check if a string is a valid number (integer or float)


def is_valid_number(s):
    try:
        float(s.strip())
        return True
    except ValueError:
        return False


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


def preprocess_meta_math_qa(df, answer_column, old_question_column, new_question_column):
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
    df = df.rename(columns={old_question_column: new_question_column})
    return df


def preprocess_mmlu(df, answer_column, old_question_column, new_question_column):
    choices_column, answer_column = answer_column

    def get_numeric_final_answer(row):
        choices = row[choices_column]
        answer_idx = row[answer_column]

        # Ensure answer_info is a string
        if not isinstance(answer_idx, int):
            return None

        final_answer = choices[answer_idx].replace(',', '.')
        if is_valid_number(final_answer):
            return final_answer
        return None

    df['numeric_final_answer'] = df.apply(get_numeric_final_answer, axis=1)
    df = df.rename(columns={old_question_column: new_question_column})
    return df


def preprocess_open_math_instruct(df, answer_column, old_question_column, new_question_column):
    result_column = answer_column

    def get_numeric_final_answer(row):
        result = row[result_column]
        result = result.replace(',', '.')
        if is_valid_number(result):
            return result
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
    answer_column = dataset_info["answer_column"]
    old_question_column = dataset_info["old_question_column"]
    new_question_column = dataset_info["new_question_column"]
    preprocess_function = dataset_info["preprocess_function"]
    # Get the config name if provided
    config_name = dataset_info.get("config_name", None)

    # Load the dataset
    if dataset_name == "nvidia/OpenMathInstruct-2":
        dataset = load_dataset(
            dataset_name, split="train_1M", trust_remote_code=True)
        combined_dataset = dataset
    else:
        if config_name:
            dataset = load_dataset(
                dataset_name, config_name, trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_name, trust_remote_code=True)
        combined_dataset = concatenate_datasets(
            [ds for split, ds in dataset.items()])

    # Convert to pandas DataFrame for easier manipulation
    df = combined_dataset.to_pandas()

    # Apply the specific preprocess function to find the numeric final answer
    df = preprocess_function(
        df, answer_column, old_question_column, new_question_column)

    # Filter out rows without a numeric final answer
    df = df.dropna(subset=['numeric_final_answer'])

    # Print the number of records before saving
    print(
        f"Number of records in {dataset_name} after preprocessing: {len(df)}")

    # Save the processed dataset in Parquet format to save disk space
    df.to_parquet(
        f"{save_path}/{dataset_name.replace('/', '_')}_processed.parquet", index=False)


if __name__ == '__main__':
    # Dictionary mapping dataset names to their specific preprocess functions and answer columns
    datasets_to_process = [
        {"dataset_name": "allenai/math_qa", "answer_column": ("options", "correct"),
         "preprocess_function": preprocess_math_qa,
         "old_question_column": "Problem",
         "new_question_column": "question",
         },

        {"dataset_name": "meta-math/MetaMathQA", "answer_column": "response",
         "preprocess_function": preprocess_meta_math_qa,
         "old_question_column": "original_question",
         "new_question_column": "question",
         },


        {"dataset_name": "cais/mmlu", "answer_column": ("choices", "answer"), "config_name": "abstract_algebra",
         "preprocess_function": preprocess_mmlu,
         "old_question_column": "question",
         "new_question_column": "question",
         },

        # {"dataset_name": "nvidia/OpenMathInstruct-2", "answer_column": "expected_answer",
        #  "preprocess_function": preprocess_open_math_instruct,
        #  "old_question_column": "problem",
        #  "new_question_column": "question", },

        {"dataset_name": "openai/gsm8k", "answer_column": "answer", "config_name": "main",
         "preprocess_function": preprocess_gsm8k,
         "old_question_column": "question",
         "new_question_column": "question",
         },

        {"dataset_name": "ChilleD/MultiArith", "answer_column": "final_ans",
         "preprocess_function": preprocess_multi_arith,
         "old_question_column": "question",
         "new_question_column": "question"
         },

    ]

    # Path to save processed datasets
    save_path = "data"

    # Process each dataset
    for dataset_info in datasets_to_process:
        process_and_save_dataset(dataset_info, save_path)
