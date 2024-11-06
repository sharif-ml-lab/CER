import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from Decoding import get_device, cot_decode
from datasets import load_dataset
from tqdm import tqdm

# Load the model and tokenizer
model_name = "/data/TensorRT-LLM/Meta-Llama-3.1-70B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, device_map='auto', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def multiarith():
    # Load the MultiArith dataset
    dataset = load_dataset("ChilleD/MultiArith")

    # Get the test set
    test_set = dataset['test']

    total_questions = len(test_set)
    correct_answers = 0

    # Iterate through each example in the test set
    with tqdm(total=total_questions, desc="Processing MultiArith", dynamic_ncols=True) as pbar:
        for idx, example in enumerate(test_set):
            question = example['question']
            correct_answer = example['final_ans']
            
            # Prepare the message for the model
            messages = [
                {"role": "user", "content": question}
            ]
            
            # Generate the response using CoT decoding
            result, confidence, final_ans = cot_decode(model, tokenizer, messages, aggregate_paths=False, max_new_tokens=512, k=10,
                                                    decoding_mode='baseline')
            
            # Compare the model's answer with the correct answer
            try:
                model_answer = float(final_ans)
                correct_answer = float(correct_answer)
                if model_answer == correct_answer:
                    correct_answers += 1
            except ValueError:
                # If parsing fails, we assume the answer is incorrect
                pass

            # Update progress bar with running accuracy
            running_accuracy = correct_answers / (idx + 1) * 100
            pbar.set_postfix(idx=idx + 1, running_accuracy=f"{running_accuracy:.2f}%")
            pbar.update(1)

    # Calculate final accuracy
    accuracy = correct_answers / total_questions * 100
    print(f"Final Accuracy: {accuracy:.2f}%")
    return accuracy

def gsm8k():
    # Load the GSM8K dataset
    dataset = load_dataset("openai/gsm8k", "main")

    # Get the test set
    test_set = dataset['test']

     # Define your desired sample size
    n = 300  # Replace with the number of samples you want to take
    seed = 11  # Define a specific seed to ensure reproducibility

    # Shuffle the dataset with a specific seed
    shuffled_dataset = test_set.shuffle(seed=seed)

    # Take the first 'n' samples
    sampled_dataset = shuffled_dataset.select(range(n))

    total_questions = len(sampled_dataset)
    correct_answers = 0

    # Iterate through each example in the test set
    with tqdm(total=total_questions, desc="Processing GSM8K", dynamic_ncols=True) as pbar:
        for idx, example in enumerate(sampled_dataset):
            question = example['question']
            correct_answer = example['answer']
            correct_answer = correct_answer.split('####')[-1]

            # Prepare the message for the model
            messages = [
                {"role": "user", "content": question}
            ]
            
            # Generate the response using CoT decoding
            result, confidence, final_ans = cot_decode(model, tokenizer, messages, aggregate_paths=False, max_new_tokens=512, k=10,
                                                    decoding_mode='baseline')
            
            # Compare the model's answer with the correct answer
            try:
                model_answer = float(final_ans)
                correct_answer = float(correct_answer)
                if model_answer == correct_answer:
                    correct_answers += 1
            except ValueError:
                # If parsing fails, we assume the answer is incorrect
                pass

            # Update progress bar with running accuracy
            running_accuracy = correct_answers / (idx + 1) * 100
            pbar.set_postfix(idx=idx + 1, running_accuracy=f"{running_accuracy:.2f}%")
            pbar.update(1)

    # Calculate final accuracy
    accuracy = correct_answers / total_questions * 100
    print(f"Final Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == '__main__':

    multiarith()
    gsm8k()
