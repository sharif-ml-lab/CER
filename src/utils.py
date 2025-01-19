import os
import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# load the model and its tokenizer
def load_model_and_tokenizer(model_name, read_model_from_huggingface=True):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='cuda',
        local_files_only=read_model_from_huggingface,
        torch_dtype=torch.bfloat16
    )
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


# load multiple parquet datasets from the given list of filenames, shuffle each dataset based on the given seed, and select up to n records.
def load_and_sample_parquet_datasets(data_dir, dataset_files, number_samples, seed):
    loaded_datasets = {}
    for filename, file_path in dataset_files.items():
        file_path = os.path.join(data_dir, file_path)
        if os.path.isfile(file_path):
            df = pd.read_parquet(file_path)
            df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
            if len(df) > number_samples:
                df = df.head(number_samples)
            loaded_datasets[filename] = df
    return loaded_datasets


def extract_final_answer(text):
    pattern = r"The final answer is ([^,\.]+)"
    match = re.search(pattern, text)
    return match.group(1) if match else None


def extract_proper_nouns(doc):
    proper_nouns = []
    current_proper_noun = []

    for token in doc:
        if token.pos_ == "PROPN":
            current_proper_noun.append(token.text)
        elif current_proper_noun and (token.like_num or "-" in token.text):
            current_proper_noun.append(token.text)
        elif current_proper_noun:
            proper_nouns.append(" ".join(current_proper_noun))
            current_proper_noun = []

    if current_proper_noun:
        proper_nouns.append(" ".join(current_proper_noun))

    return proper_nouns


# construct prompt for given question


def construct_prompt(question, few_shot=True, few_shot_path=None, multihop=False):
    if few_shot:  # few-shot setting
        few_shots = read_from_txt(few_shot_path)
        base_prompt = few_shots.format(question=question)
    else:  # zero-shot setting
        if not multihop:
            base_prompt = f"Q: {question}\nA: Let's think step by step.\n Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
        else:
            base_prompt = f"Q: {question}\nA: Let's Solve step by step.\n focusing only on the essential steps and limiting your response to 5 sentences. Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
    return base_prompt


def p_true_second_query_construct_prompt(question, answer):
    base_prompt = f"Please answer either with ‘True’ or ‘False’ only. Is it True that: {question} {answer}"
    return base_prompt


def postprocess_final_answer(numeric_expression: str) -> str:
    try:
        cleaned_up = numeric_expression.replace(',', '')
        result = eval(cleaned_up)
        return str(result)
    except Exception:
        print(f'can not clean this value {numeric_expression}:')
        return numeric_expression


# extract the last numerical value from a given text
def extract_last_numerical_value(text):
    matches = re.findall(r'([+-]?\d?[0-9.,/*\-+]*\d)', text)
    return matches[-1] if matches else None


# extract all numerical values from a given text
def extract_all_numerical_values(text):
    return re.findall(r'([+-]?\d?[0-9.,/*\-+]*\d)', text)


# write content in a file_path


def write_to_txt(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


# read content from a file_path


def read_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


# save results in a csv file
def save_results_to_csv(results, filename):
    results_df = pd.DataFrame(results)
    results_df.to_csv(filename, index=False)


# print the final accuracy
def print_final_accuracy(description, accuracy):
    print(f"Final Accuracy for {description}: {accuracy:.2f}%")
