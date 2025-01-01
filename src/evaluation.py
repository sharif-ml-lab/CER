from tqdm import tqdm

from src.self_consistency import self_consistency_decode
from src.decoding import cot_decode
from src.greedy_on_numbers import greedy_number_cot_decode
from src.utils import load_and_sample_dataset, load_model_and_tokenizer, construct_prompt, print_final_accuracy, save_results_to_csv
from src.config import Config

# evaluate the model on a single example
def evaluate_single_example(
        model,
        tokenizer,
        question,
        correct_answer_str,
        k,
        aggregate,
        decoding_mode,
        scoring_mode,
        baseline_cot,
        sampling_mode,
        few_shot,
        few_shot_path,
        dataset_desc,
        confidence_method):
    
    messages = [{"role": "user", "content": construct_prompt(question=question, few_shot=few_shot, few_shot_path=few_shot_path)}]


    # pick k-branch and continue each path with greedy sampling.
    if baseline_cot == "k-branch" or baseline_cot == "k-seperate":
        result, confidence, final_ans = cot_decode(
            model,
            tokenizer,
            messages,
            aggregate_paths=aggregate,
            k=k,
            decoding_mode=decoding_mode,
            sampling_mode=sampling_mode,
            scoring_mode=scoring_mode,
            baseline_cot=baseline_cot,
            confidence_method=confidence_method,
        )

    # elif baseline_cot == "greedy_decoding": # ??????????????????
    #     result, confidence, final_ans = greedy_number_cot_decode( # greedy
    #         model,
    #         tokenizer,
    #         messages,
    #         aggregate_paths=aggregate,
    #         k=k,
    #         sampling_mode=sampling_mode,
    #         scoring_mode=scoring_mode
    #     )
        
    elif baseline_cot == "self_consistency":
        result, confidence, final_ans = self_consistency_decode(model, tokenizer, messages,k=k) 

    try:
        model_answer = float(final_ans)
        correct_answer = float(correct_answer_str)
        is_correct = (model_answer == correct_answer)
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


# evaluate the model on dataset
def evaluate_dataset(model, tokenizer, dataset, k, aggregate, decoding_mode, description, scoring_mode, baseline_cot, sampling_mode, few_shot, few_shot_path, confidence_method):
    total_questions = len(dataset)
    correct_answers = 0
    results = []

    with tqdm(total=total_questions, desc=f"Processing {description}", dynamic_ncols=True) as pbar:
        for idx, example in enumerate(dataset):
            question = example['question']
            if 'final_ans' in example:
                correct_answer = example['final_ans']
            else:
                correct_answer = example['answer'].split('####')[-1]

            result_dict = evaluate_single_example(model, tokenizer, question, correct_answer, k, aggregate, decoding_mode, scoring_mode, baseline_cot, sampling_mode, few_shot, few_shot_path, description, confidence_method)
            results.append(result_dict)

            if result_dict['is_correct']:
                correct_answers += 1

            running_accuracy = (correct_answers / (idx + 1)) * 100
            pbar.set_postfix(idx=idx + 1, running_accuracy=f"{running_accuracy:.2f}%")
            pbar.update(1)

    save_results_to_csv(results, f"{description}_evaluation_results.csv")
    accuracy = (correct_answers / total_questions) * 100
    print_final_accuracy(description, accuracy)
    return accuracy



def run_dataset(config: Config):
    model_name=config.model_name
    dataset_desc=config.dataset_desc
    decoding_mode=config.decoding_mode
    baseline_cot=config.baseline_cot
    scoring_mode=config.scoring_mode
    aggregate=config.aggregate
    K=config.K
    sampling_mode=config.sampling_mode
    few_shot=config.few_shot
    confidence_method= config.confidence
    
    model, tokenizer = load_model_and_tokenizer(model_name)

    information = f"model: {model_name}\n decoding mode: {decoding_mode}\n K: {K}\n aggregate: {aggregate}\n sampling mode: {sampling_mode}\n baseline cot: {baseline_cot}\n scoring mode: {scoring_mode}\n few-shot: {few_shot}\n confidence-method: {confidence_method}\n"
    print(information)

    if dataset_desc == "MultiArith":
        if few_shot: few_shot_path = config.multiarith_shots
        multiarith_dataset = load_and_sample_dataset(dataset_name="ChilleD/MultiArith", split="test")
        evaluate_dataset(
            model, tokenizer, multiarith_dataset,
            k=K,
            aggregate=aggregate,
            decoding_mode=decoding_mode,
            description="MultiArith",
            scoring_mode=scoring_mode,
            baseline_cot=baseline_cot,
            sampling_mode=sampling_mode,
            few_shot = few_shot,
            few_shot_path = few_shot_path,
            confidence_method=confidence_method,
        )

    elif dataset_desc == "GSM8K":
        if few_shot: few_shot_path = config.gsm8k_shots
        gsm8k_dataset = load_and_sample_dataset(dataset_name="openai/gsm8k", split='main', subset="train", sample_size=300, seed=11)
        evaluate_dataset(
            model, tokenizer, gsm8k_dataset,
            k=K,
            aggregate=aggregate,
            decoding_mode=decoding_mode,
            description="GSM8K",
            scoring_mode=scoring_mode,
            baseline_cot=baseline_cot,
            sampling_mode=sampling_mode,
            few_shot = few_shot,
            few_shot_path = few_shot_path,
            confidence_method=confidence_method,
        )
