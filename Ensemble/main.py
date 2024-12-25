import numpy as np
from datasets import load_dataset
from lm_wrapper import LanguageModelWrapper
from aggregator import MaxLogProbAggregator
from tqdm import tqdm


def construct_prompt(question: str) -> str:
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


def load_and_sample_dataset(dataset_name: str, split: str, subset: str = None, sample_size: int = None,
                            seed: int = None):
    """
    Load a dataset and optionally sample from it.
    """
    if subset is None:
        dataset = load_dataset(dataset_name, split=split)
    else:
        dataset = load_dataset(dataset_name, split)[subset]
    if sample_size is not None and seed is not None:
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))
    return dataset


def evaluate_single_example(
        models, aggregator, question, correct_answer,
) -> dict:
    """
    Evaluate the model on a single example.
    """
    messages = [{"role": "user", "content": construct_prompt(question)}]

    # CoT prompt example:
    prompt = construct_prompt(question)

    # Gather answers from each model
    answers_with_probs = []
    for lm in models:
        output = lm.generate_answer(prompt)
        answers_with_probs.append(output)

    # Aggregate the answers to get a final prediction
    final_answer = aggregator.aggregate(answers_with_probs)

    # Check correctness
    try:
        model_answer = float(final_answer)
        correct_answer = float(correct_answer)
        is_correct = np.abs((model_answer - correct_answer)) <= 1e-10
    except ValueError:
        is_correct = False

    return {
        'is_correct': is_correct
    }


def main(dataset, models, aggregator):
    total_questions = len(dataset)
    correct_answers = 0

    with tqdm(total=total_questions, desc=f"Processing...", dynamic_ncols=True) as pbar:
        for idx, example in enumerate(dataset):
            question = example['question']
            if 'final_ans' in example:
                correct_answer = example['final_ans']
            else:
                correct_answer = example['answer'].split('####')[-1]

            result_dict = evaluate_single_example(
                models, aggregator, question, correct_answer,
            )

            if result_dict['is_correct']:
                correct_answers += 1

            running_accuracy = (correct_answers / (idx + 1)) * 100
            pbar.set_postfix(idx=idx + 1, running_accuracy=f"{running_accuracy:.2f}%")
            pbar.update(1)

    accuracy = (correct_answers / total_questions) * 100
    print(f"Accuracy on the subset of the dataset: {accuracy:.2f}")
    return accuracy


if __name__ == "__main__":
    # 2. Define a list of model names to load (1B - 10B range are examples)
    model_names = [
        "EleutherAI/gpt-neo-1.3B",  # ~1.3B
        "EleutherAI/gpt-neo-2.7B",  # ~2.7B
        # "facebook/opt-6.7b",       # ~6.7B (example of a bigger model)
        "facebook/opt-2.7b",
        # Add or remove as desired
    ]

    # 3. Instantiate LanguageModelWrapper for each model
    device = 'cuda'  # or 'cuda' if you have a GPU
    models = []
    for model_name in model_names:
        lm = LanguageModelWrapper(model_name, device=device)
        models.append(lm)

    # 4. Define aggregator
    #    We can switch out aggregator classes or implement your own aggregator logic.
    #    For instance: aggregator = MajorityVoteAggregator()
    aggregator = MaxLogProbAggregator()

    multiarith_dataset = load_and_sample_dataset("ChilleD/MultiArith", "test")
    main(multiarith_dataset, models=models, aggregator=aggregator)

    # Evaluate GSM8K dataset (sample of 300)
    gsm8k_dataset = load_and_sample_dataset("openai/gsm8k", split='main', subset="train", sample_size=300, seed=11)
    main(gsm8k_dataset, models=models, aggregator=aggregator)
