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
    for idx, example in tqdm(enumerate(test_set)):
        question = example['question']
        correct_answer = example['final_ans']
        
        # Prepare the message for the model
        messages = [
            {"role": "user", "content": question}
        ]
        
        # Generate the response using CoT decoding
        print(f"Processing question {idx + 1}/{total_questions}...")
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

    # Calculate accuracy
    accuracy = correct_answers / total_questions * 100
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

def gsm8k():
    # Load the MultiArith dataset
    dataset = load_dataset("openai/gsm8k", "main")

    # Get the test set
    test_set = dataset['test']

    total_questions = len(test_set)
    correct_answers = 0

    # Iterate through each example in the test set
    for idx, example in tqdm(enumerate(test_set)):
        question = example['question']
        correct_answer = example['answer']
        correct_answer = correct_answer.split('####')[-1]
        
        # Prepare the message for the model
        messages = [
            {"role": "user", "content": question}
        ]
        
        # Generate the response using CoT decoding
        print(f"Processing question {idx + 1}/{total_questions}...")
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

    # Calculate accuracy
    accuracy = correct_answers / total_questions * 100
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == '__main__':

    messages = [
    {"role": "user",
     "content": "In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What percentage of the entire students enrolled in hip-hop dance?"}
    ]

    # Generate the response using CoT decoding
    print(f"Using device: {get_device()}")
    result, confidence, final_ans = cot_decode(model, tokenizer, messages, aggregate_paths=True, max_new_tokens=512, k=2,
                                            decoding_mode='baseline')
    print(f"CoT Decoding:\n {final_ans}")

