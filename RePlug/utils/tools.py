import numpy as np


def ensemble_aggregate(probs, scores):
    weights = scores.squeeze(0)
    # Broadcasting weights to match each tensor in probs and summing them
    weighted_sum = np.sum([weight * prob for weight, prob in zip(weights, probs)], axis=0)

    return weighted_sum


def my_aggregate(probs, scores):
    weights = scores.squeeze(0)
    # Broadcasting weights to match each tensor in probs and summing them
    weighted_sum = np.sum([np.log(weight * prob) for weight, prob in zip(weights, probs)], axis=0)

    return weighted_sum


def greedy_aggregate(probs, scores):
    return probs[0]


def aggregate_token_probs(probs, scores, mode):
    if mode == 'ensemble':
        return ensemble_aggregate(probs, scores)
    elif mode == 'new':
        return my_aggregate(probs, scores)
    elif mode == 'greedy':
        return greedy_aggregate(probs, scores)

    raise NotImplementedError()


def fill_template(question, doc, choices):
    base_nq_template = f"""Knowledge: received 122,000 buys (excluding WWE Network views), down from the previous yearś 199,000 buys. The event is named
after the Money In The Bank ladder match, in which multiple wrestlers use ladders to retrieve a briefcase hanging above the ring. The
winner is guaranteed a match for the WWE World Heavyweight Championship at a time of their choosing within the next year. On the
June 2 episode of "Raw", Alberto Del Rio qualified for the match by defeating Dolph Ziggler. The following week, following Daniel
Bryan being stripped of his WWE World Championship due to injury, Stephanie McMahon changed the
Question: Who won the mens money in the bank match?
Answer: Braun Strowman
Knowledge: in 3D on March 17, 2017. The first official presentation of the film took place at Disneyś three-day D23 Expo in August
2015. The world premiere of "Beauty and the Beast" took place at Spencer House in London, England on February 23, 2017; and the
film later premiered at the El Capitan Theatre in Hollywood, California, on March 2, 2017. The stream was broadcast onto YouTube. A
sing along version of the film released in over 1,200 US theaters nationwide on April 7, 2017. The United Kingdom received the same
version on April 21, 2017. The film was re-released in
Question: When does beaty and the beast take place
Answer: Rococo-era
Knowledge:{doc}
Question: {question}"""
    base_mmlu_template = f"""Answer the following questions by choosing the correct option (A, B, C, or D). Provide your answer as a single character.

Example 1:
Knowledge: Although over half of Europe’s original forests disappeared through the centuries of deforestation, Europe still has over one quarter of its land area as forest, such as the broadleaf and mixed forests, taiga of Scandinavia and Russia, mixed rainforests of the Caucasus and the Cork oak forests in the western Mediterranean. During recent times, deforestation has been slowed and many trees have been planted. However, in many cases monoculture plantations of conifers have replaced the original mixed natural forest, because these grow quicker.

Question: As of 2015, since 1990 forests have ____ in Europe and have ____ in Africa and the Americas.

A. "increased, increased"  B. "increased, decreased"  C. "decreased, increased"  D. "decreased, decreased"

Answer: B

Example 2:
Knowledge: Over the past decades, the political outlook of Americans has become more progressive, with those below the age of thirty being considerably more liberal than the overall population. According to recent polls, 56% of those age 18 to 29 favor gay marriage, 68% state environmental protection to be as important as job creation, 52% "think immigrants śtrengthen the country with their hard work and talents,"´ 62% favor a "tax financed, government-administrated universal health care" program and 74% "say ṕeopleś willśhould have more influence on U.S. laws than the Bible, compared to 37%, 49%, 38%, 47% and 58% among the

Question: As of 2019, about what percentage of Americans agree that the state is run for the benefit of all the people?

A. 31% B. 46% C. 61% D. 76%

Answer: B

Example 3:
Knowledge: The mitochondria are known as the powerhouses of the cell. They generate the energy that cells need to function.

Question: What is the main function of mitochondria in the cell?

A. Protein synthesis  B. Waste disposal  C. DNA replication  D. Energy production

Answer: D

Example 4:
Knowledge: The capital of France is Paris, which is also the most populous city in the country.

Question: What is the capital city of France?

A. Madrid  B. Berlin  C. Paris  D. Rome

Answer: C

Here is the example that you have to solve.

Knowledge:{doc}

Question: {question}

A. {choices[0]} B. {choices[1]} C. {choices[2]} D. {choices[3]}

Answer: """

    return base_mmlu_template


def construct_prompt(question, doc, choices, client):
    messages = fill_template(question, doc, choices)

    # Use the chat template to format the input
    if client.tokenizer.chat_template:
        chat_input = [{"role": "user", "content": messages}]
        input_text = client.tokenizer.apply_chat_template(chat_input, tokenize=False, add_generation_prompt=True)
        return input_text
    else:
        # Fallback for tokenizers without chat templates
        return messages


def mmlu_pred_checker(final_result_text, answer):
    # Define a dictionary to map the answer index to the character (A, B, C, D)
    answer_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}

    # Get the corresponding character for the answer
    correct_answer_char = answer_map.get(answer, '')

    # Check if the correct answer character is present in the final_result_text
    if correct_answer_char == final_result_text.lower().strip():
        return True

    return False


def prediction_is_correct(final_result_text, answer, mode):
    if mode == 'mmlu':
        return mmlu_pred_checker(final_result_text, answer)


def calculate_doc_scores(raw_distances):
    softmax_scores = np.exp(raw_distances) / np.sum(np.exp(raw_distances))
    return softmax_scores


def calculate_perplexity(log_probs):
    """
    Calculate perplexity given a list of log probabilities.
    """
    return np.exp(-np.mean(log_probs))


def evaluate_retrieval(retrieved_docs, ground_truth):
    """
    Simple evaluation of the retriever based on overlap with ground truth.

    Args:
        retrieved_docs (list): List of retrieved document IDs.
        ground_truth (list): List of ground truth document IDs.

    Returns:
        float: Precision metric for retrieved documents.
    """
    retrieved_set = set(retrieved_docs)
    ground_truth_set = set(ground_truth)
    if len(retrieved_set) == 0:
        return 0.0
    return len(retrieved_set & ground_truth_set) / len(retrieved_set)


def evaluate_language_model(predictions, targets):
    """
    Evaluate LM output against a list of ground-truth sequences using accuracy.

    Args:
        predictions (list of str): List of predictions generated by the language model.
        targets (list of str): List of target ground-truth sequences.

    Returns:
        float: Average accuracy of the model.
    """
    correct = sum([1 for pred, target in zip(predictions, targets) if pred.strip() == target.strip()])
    return correct / len(targets) if targets else 0.0
