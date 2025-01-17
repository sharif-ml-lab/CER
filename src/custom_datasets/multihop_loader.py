from datasets import load_dataset
import pandas as pd

dataset_name = "hotpot"

if dataset_name == "hotpot":
    # dict_keys(['supporting_facts', 'level', 'question', 'context', 'answer', '_id', 'type']) 90447
    dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split='test', trust_remote_code=True)

    dataset = dataset[dataset['type'] != "comparison"]
    dataset = dataset[['question', 'answer', 'level']]

    df = dataset.dropna(subset=['answer'])
    df.to_parquet(
        f"data/hotpotqa_processed.parquet", index=False)

elif dataset_name == "trivia":
    # train: 61888; validation: 7993; test: 7701
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext")
    dataset = dataset['validation'].to_pandas()  # train/validation/test
    dataset = dataset[['question', 'answer']]
    dataset['answer'] = dataset['answer'].apply(
        lambda x: x['normalized_aliases'] if 'normalized_aliases' in x else None)

    df = dataset.dropna(subset=['answer'])
    df.to_parquet(
        f"data/triviaqa_processed.parquet", index=False)
