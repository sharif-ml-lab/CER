import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from Decoding import cot_decode
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from self_consistency import self_consistency_decode


def load_model_and_tokenizer(model_name):
    """
    Load the model and tokenizer from the specified path.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name, local_files_only=True, device_map='cuda', torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def construct_prompt(question):
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


def evaluate_dataset(model, tokenizer, dataset, k, aggregate, decoding_mode, description, scoring_mode, COT=False):
    """
    Evaluate the model on the given dataset.
    """
    total_questions = len(dataset)
    correct_answers = 0
    results = []

    with tqdm(total=total_questions, desc=f"Processing {description}", dynamic_ncols=True) as pbar:
        for idx, example in enumerate(dataset):
            question = example['question']
            correct_answer = example['final_ans'] if 'final_ans' in example else example['answer'].split('####')[-1]

            # Prepare the message for the model
            messages = [
                {"role": "user", "content": construct_prompt(question)}
            ]

            if COT:
                # Generate the response using CoT decoding
                result, confidence, final_ans = cot_decode(
                    model, tokenizer, messages, aggregate_paths=aggregate, max_new_tokens=512, k=k,
                    decoding_mode=decoding_mode, sampling_mode="temp", scoring_mode=scoring_mode
                )
            else:
                result, confidence, final_ans = self_consistency_decode(
                    model, tokenizer, messages, aggregate_paths=aggregate, k=k,
                    decoding_mode=decoding_mode
                )

            # Compare the model's answer with the correct answer
            try:
                model_answer = float(final_ans)
                correct_answer = float(correct_answer)
                is_correct = model_answer == correct_answer
                if is_correct:
                    correct_answers += 1
            except ValueError:
                # If parsing fails, we assume the answer is incorrect
                is_correct = False

            # Save the result
            results.append({
                'question': question,
                'correct_answer': correct_answer,
                'predicted_answer': result,
                'predicted_final_answer': final_ans,
                'confidence_score': confidence,
                'is_correct': is_correct
            })

            # Update progress bar with running accuracy
            running_accuracy = correct_answers / (idx + 1) * 100
            pbar.set_postfix(idx=idx + 1, running_accuracy=f"{running_accuracy:.2f}%")
            pbar.update(1)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{description}_evaluation_results.csv", index=False)

    # Calculate final accuracy
    accuracy = correct_answers / total_questions * 100
    print(f"Final Accuracy for {description}: {accuracy:.2f}%")
    return accuracy


def load_and_sample_dataset(dataset_name, split, sample_size=None, seed=None):
    """
    Load a dataset and optionally sample from it.
    """
    dataset = load_dataset(dataset_name, split=split)
    if sample_size is not None and seed is not None:
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))
    return dataset


if __name__ == '__main__':
    # Configurations
    model_name = "/data/models/Meta-Llama-3.1-8B-Instruct"
    K = 10
    AGGREGATE = False
    DECODING_MODE = 'new'
    BASELINE_COT = True
    # scoring_mode = 'min'
    # scoring_mode = 'log'
    # scoring_mode = 'h_mean'
    scoring_mode = 'max'
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    print(model_name)
    print(f'Mode: CoT + {DECODING_MODE}')
    print(f'Config: k = {K}, Aggregate = {AGGREGATE}, scoring_mode = {scoring_mode}')

    # Evaluate MultiArith dataset
    multiarith_dataset = load_and_sample_dataset("ChilleD/MultiArith", "test")
    evaluate_dataset(model, tokenizer, multiarith_dataset, k=K, aggregate=AGGREGATE, decoding_mode=DECODING_MODE,
                     description="MultiArith", scoring_mode=scoring_mode, COT=BASELINE_COT)

    # Evaluate GSM8K dataset (with sampling)
    gsm8k_dataset = load_and_sample_dataset("openai/gsm8k", "test", sample_size=300, seed=11)
    evaluate_dataset(model, tokenizer, gsm8k_dataset, k=K, aggregate=AGGREGATE, decoding_mode=DECODING_MODE,
                     description="GSM8K", scoring_mode=scoring_mode, COT=BASELINE_COT)
