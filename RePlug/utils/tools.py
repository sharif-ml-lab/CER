import numpy as np


def ensemble_aggregate(probs, scores):
    weights = scores.squeeze(0)
    # Broadcasting weights to match each tensor in probs and summing them
    weighted_sum = np.sum([weight * prob for weight, prob in zip(weights, probs)], axis=0)

    return weighted_sum


def my_aggregate(probs, scores):
    pass


def aggregate_token_probs(probs, scores, mode):
    if mode == 'ensemble':
        return ensemble_aggregate(probs, scores)
    elif mode == 'new':
        return my_aggregate(probs, scores)

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

    base_mmlu_template = f"""Knowledge: Arctic Ocean. Although over half of Europe’s original forests disappeared through the centuries of deforestation, Europe
still has over one quarter of its land area as forest, such as the broadleaf and mixed forests, taiga of Scandinavia and Russia, mixed
rainforests of the Caucasus and the Cork oak forests in the western Mediterranean. During recent times, deforestation has been slowed
and many trees have been planted. However, in many cases monoculture plantations of conifers have replaced the original mixed natural
forest, because these grow quicker. The plantations now cover vast areas of land, but offer poorer habitats for many European
Question: As of 2015, since 1990 forests have
in Europe and have
in Africa and the Americas.
A. "increased, increased" B. "increased, decreased" C. "decreased, increased" D. "decreased, decreased"
Answer: B

Knowledge: Over the past decades, the political outlook of Americans has become more progressive, with those below the age of thirty
being considerably more liberal than the overall population. According to recent polls, 56% of those age 18 to 29 favor gay marriage,
68% state environmental protection to be as important as job creation, 52% "think immigrants śtrengthen the country with their hard
work and talents,"´ 62% favor a "tax financed, government-administrated universal health care" program and 74% "say ṕeopleś willśhould
have more influence on U.S. laws than the Bible, compared to 37%, 49%, 38%, 47% and 58% among the
Question: As of 2019, about what percentage of Americans agree that the state is run for the benefit of all the people?
A. 31% B. 46% C. 61% D. 76%
Answer: B

Knowledge: last week at a United Nations climate meeting in Germany, China and India should easily exceed the targets they set for
themselves in the 2015 Paris Agreement... India is now expected to obtain 40 percent of its electricity from non-fossil fuel sources by
2022, eight years ahead of schedule." Solar power in Japan has been expanding since the late 1990s. By the end of 2017, cumulative
installed PV capacity reached over 50 GW with nearly 8 GW installed in the year 2017. The country is a leading manufacturer of solar
panels and is in the top 4 ranking for countries
Question: Which of the following countries generated the most total energy from solar sources in 2019?
A. China B. United States C. Germany D. Japan
Answer: D

Knowledge:{doc}
Question: {question}
A. {choices[0]} B. {choices[1]} C. {choices[2]} D. {choices[3]}"""

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


def prediction_is_correct(final_result_text, answer, param):
    # if final_result_text.lower():
    pass


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
