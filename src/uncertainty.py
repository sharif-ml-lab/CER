import random
import json
import re

import torch
import numpy as np

from transformers import PreTrainedModel, PreTrainedTokenizer
from src.utils import extract_all_numerical_values, extract_final_answer, extract_last_numerical_value, extract_proper_nouns, postprocess_final_answer, extract_all_steps, extract_last_proper_noun


def _handle_last_decoding(
        tokenizer: PreTrainedTokenizer,
        device,
        answer_text,
        output_scores,
        answer_ids,
        confidence_method,
        multihop,
        doc,
):
    if not multihop:
        final_answer = extract_last_numerical_value(answer_text)
    else:
        all_values = extract_proper_nouns(doc)
        final_answer = extract_final_answer(answer_text)

        if not final_answer and not all_values:
            return None

        final_answer = final_answer if final_answer else all_values[-1]

    if final_answer is None:
        return None

    final_answer_ids = tokenizer.encode(final_answer, add_special_tokens=False)
    answer_ids_list = answer_ids.tolist()

    final_answer_start_idx = _find_subsequence_indices(
        answer_ids_list, final_answer_ids, 1)
    if final_answer_start_idx == -1:  # second try
        final_answer_ids = tokenizer.encode(
            " " + final_answer, add_special_tokens=False)
        final_answer_start_idx = _find_subsequence_indices(
            answer_ids_list, final_answer_ids, 1)

        if final_answer_start_idx == -1:
            return None
    final_answer_start_idx -= 1

    if final_answer_start_idx < 0 or final_answer_start_idx + len(final_answer_ids) > len(output_scores):
        return None

    final_answer_scores = output_scores[final_answer_start_idx:
                                        final_answer_start_idx + len(final_answer_ids)]
    confidence = calculate_confidence_for_final_answer(tokenizer, final_answer_scores,
                                                       torch.tensor(final_answer_ids, device=device), confidence_method)

    if not multihop:
        final_answer = postprocess_final_answer(final_answer)

    return answer_text, confidence, final_answer


def number_step_by_step_extraction(answer_text, step_decomposition):
    # step by step extracting
    if step_decomposition:
        steps = extract_all_steps(answer_text)
        seen_dict = {}
        all_values = [extract_all_numerical_values(
            step) for step in steps]
        new_all_values = []
        for all_value in all_values:
            for num_value in all_value:
                seen_dict[num_value] = seen_dict.get(num_value, 0) + 1
            new_all_values.append(
                (seen_dict[all_value[-1]], all_value[-1]))
        all_values = new_all_values.copy()

        if len(extract_all_numerical_values(answer_text)) > 0:
            final_answer = extract_all_numerical_values(
                answer_text)[-1]
        else:
            final_answer = None

        if not all_values:
            step_decomposition = False
            all_values = extract_all_numerical_values(answer_text)

    else:
        all_values = extract_all_numerical_values(answer_text)

    return final_answer, all_values, step_decomposition


def proper_noun_step_by_step_extraction(answer_text, nlp, step_decomposition, doc):
    if step_decomposition:
        # check the period and split based on this.
        steps = re.split(r"\.\s+", answer_text)
        steps = [step for step in steps if len(step)]

        # steps = extract_all_steps(answer_text)

        doc_steps = list(nlp.pipe(steps))
        seen_dict = {}
        all_values = [extract_proper_nouns(
            doc_step) for doc_step in doc_steps]
        final_answer = extract_final_answer(answer_text)

        if final_answer:
            all_values.append([final_answer])

        new_all_values = []
        for all_value in all_values:
            for proper_noun_value in all_value:
                seen_dict[proper_noun_value] = seen_dict.get(
                    proper_noun_value, 0) + 1
            if len(all_value) > 0:
                new_all_values.append(
                    (seen_dict[all_value[-1]], all_value[-1]))
        all_values = new_all_values.copy()

        # print(all_values)

        if not all_values:
            step_decomposition = False
            all_values = extract_proper_nouns(doc)

    else:
        all_values = extract_proper_nouns(doc)
        final_answer = extract_final_answer(answer_text)

    return final_answer, all_values, step_decomposition


def _handle_all_decoding(
        tokenizer: PreTrainedTokenizer,
        device,
        answer_text,
        output_scores,
        answer_ids,
        scoring_mode,
        confidence_method,
        multihop, doc,
        random_selection,
        random_selection_number_words,
        step_decomposition,
        nlp,):

    if not random_selection:
        if not multihop:
            final_answer, all_values, step_decomposition = number_step_by_step_extraction(
                answer_text, step_decomposition)
        else:
            # step by step extracting
            final_answer, all_values, step_decomposition = proper_noun_step_by_step_extraction(
                answer_text, nlp, step_decomposition, doc)

            if not all_values and not final_answer:
                return None

            if not all_values and final_answer:
                all_values.append(final_answer)

            elif not step_decomposition and all_values[-1] != final_answer and final_answer:
                all_values.append(final_answer)
    else:
        step_decomposition = False
        all_values = random.sample(
            answer_text.split(), min(random_selection_number_words, len(answer_text.split())))
        if not multihop:
            final_answer = extract_last_numerical_value(answer_text)
        else:
            final_answer = extract_final_answer(answer_text)

        if all_values[-1] != final_answer and final_answer:
            all_values.append(final_answer)

    if not all_values:
        return None

    answer_ids_list = answer_ids.tolist()
    confidence_sum = 0.0
    min_conf = 10
    max_conf = -1
    total_valid_values = 0
    seen_dict = {}

    for num_idx, num_value in enumerate(all_values):

        if step_decomposition:
            occurrence_count, num_value = num_value
        else:
            seen_dict[num_value] = seen_dict.get(num_value, 0) + 1
            occurrence_count = seen_dict[num_value]

        num_value_ids = tokenizer.encode(num_value, add_special_tokens=False)

        value_start_idx = _find_subsequence_indices(
            answer_ids_list, num_value_ids, occurrence_count)

        if value_start_idx == -1:
            # next try with considering whitespace at start
            num_value_ids = tokenizer.encode(
                " " + num_value, add_special_tokens=False)

            value_start_idx = _find_subsequence_indices(
                answer_ids_list, num_value_ids, occurrence_count)

            if value_start_idx == -1:
                continue

        if value_start_idx < 0 or value_start_idx + len(num_value_ids) > len(output_scores):
            continue

        num_value_scores = output_scores[value_start_idx:
                                         value_start_idx + len(num_value_ids)]
        conf_val = calculate_confidence_for_final_answer(tokenizer, num_value_scores, torch.tensor(num_value_ids, device=device),
                                                         confidence_method)

        if scoring_mode == 'log':  # (lop(1 + c1) + ... + log(1 + cn)) / n
            confidence_sum += np.log(1 + conf_val)

        elif scoring_mode == 'min':  # min(c1, ..., cn)
            if conf_val < min_conf:
                min_conf = conf_val
                confidence_sum = conf_val

        elif scoring_mode == 'max':  # max(c1, ..., cn)
            if conf_val > max_conf:
                max_conf = conf_val
                confidence_sum = conf_val

        elif scoring_mode == 'h_mean':  # n / (1/c1 + ... + 1/cn)
            confidence_sum += 1 / (1e-11 + conf_val)

        elif scoring_mode == "mean":  # (c1 + ... + cn) / n
            confidence_sum += conf_val

        # (1*c1 + ... n*cn) / (1 + ... + n)
        elif scoring_mode == "weighted_mean":
            confidence_sum += (((1 + num_idx) * conf_val) /
                               ((len(all_values) * len(all_values)) / 2))
        elif scoring_mode == "weighted_half":  # half: final answer, half: all the others
            total = len(all_values)
            if total - 1 != num_idx:
                confidence_sum += conf_val  # coeff=1 for others
            else:
                # coeff=total-1 for the last
                confidence_sum += (total - 1) * \
                    conf_val if total > 1 else conf_val

        # (2^0*c1 + ... 2^(n-1)*cn) / (2^n - )
        elif scoring_mode == "weighted_2":
            confidence_sum += (2**num_idx) * conf_val

        else:
            raise NotImplementedError("Unsupported scoring_mode")

        total_valid_values += 1

    if total_valid_values > 0:
        if scoring_mode == 'log':
            confidence = confidence_sum.item() / total_valid_values
        elif scoring_mode in ["mean"]:
            confidence = confidence_sum / total_valid_values
        elif scoring_mode in ['min', 'max', "weighted_mean"]:
            confidence = confidence_sum
        elif scoring_mode == 'h_mean':
            confidence = total_valid_values / confidence_sum
        elif scoring_mode == "weighted_half":
            confidence = confidence_sum / \
                (2 * (len(all_values) - 1)) if len(all_values) > 1 else confidence_sum
        elif scoring_mode == "weighted_2":
            confidence = confidence_sum / ((2 ** len(all_values)) - 1)
        else:
            raise NotImplementedError

        if not multihop:
            if not step_decomposition:
                final_answer = postprocess_final_answer(all_values[-1])
            else:
                final_answer = postprocess_final_answer(final_answer)
        else:
            # previous method
            if not step_decomposition:
                final_answer = all_values[-1]

            else:
                final_answer = all_values[-1][-1]

            # for testing
            # print(final_answer)

        return answer_text, confidence, final_answer

    return None

#  compute the confidence for a single numerical value occurrence in the generated answer.


def compute_confidence_for_value(tokenizer, output_scores, answer_ids, value_ids, device, confidence_method):
    value_start_idx = _find_subsequence_indices(answer_ids, value_ids, 1)
    if value_start_idx == -1:
        return None

    value_start_idx -= 1
    if value_start_idx < 0 or value_start_idx + len(value_ids) > len(output_scores):
        return None

    value_scores = output_scores[value_start_idx:
                                 value_start_idx + len(value_ids)]
    return calculate_confidence_for_final_answer(tokenizer, value_scores, torch.tensor(value_ids, device=device), confidence_method)


# calculate the confidence score (Δ) for the final answer tokens.
def calculate_confidence_for_final_answer(tokenizer, logits, answer_ids, confidence_method: str = "default"):
    confidence_sum = 1.0
    valid_tokens = 0

    for t, token_id in enumerate(answer_ids):

        if t >= len(logits):
            break
        token_logits = logits[t]

        probs = torch.softmax(token_logits, dim=-1)
        ans_token_prob = probs[token_id]

        if confidence_method == "default":
            confidence_sum *= ans_token_prob.item()
        elif confidence_method == "sum":
            confidence_sum += ans_token_prob.item()
        elif confidence_method == "entropy":
            probs = torch.clamp(probs, min=1e-12)
            entropy = - (probs * torch.log(probs)).sum(dim=-1)
            confidence_sum = 1 - entropy.item()
        elif confidence_method == "top_2_diff_weighted":
            top_2_probs, _ = torch.topk(probs, min(2, probs.size(-1)))
            if top_2_probs.size(-1) > 1:
                confidence_sum += (top_2_probs[0] - top_2_probs[1]
                                   ).item() * ans_token_prob.item()
            else:
                confidence_sum += 1.0
        elif confidence_method == "top_2_diff":
            top_2_probs, _ = torch.topk(probs, min(2, probs.size(-1)))
            if top_2_probs.size(-1) > 1:
                confidence_sum += (top_2_probs[0] - top_2_probs[1]).item()
            else:
                confidence_sum += 1.0
        elif confidence_method == "top_2_diff_extension":
            top_20_probs, top_20_indices = torch.topk(
                probs, min(20, probs.size(-1)))
            top_token_strs = [tokenizer.decode(
                token_id, skip_special_tokens=True) for token_id in top_20_indices]
            print(top_token_strs)
            confidence_sum += (top_20_probs[0] - top_20_probs[1]).item()
        else:
            raise NotImplementedError(
                "Unsupported confidence calculation mode")

        valid_tokens += 1

    return confidence_sum / valid_tokens if valid_tokens > 0 else 0.0

#  find the start index of the Nth occurrence (occurrence_count) of subsequence in sequence. Returns -1 if not found.


def _find_subsequence_indices(sequence, subsequence, occurrence_count: int = 1):
    found_count = 0
    seq_len = len(sequence)
    sub_len = len(subsequence)

    for i in range(seq_len - sub_len + 1):
        if sequence[i:i + sub_len] == subsequence:
            if found_count == occurrence_count - 1:
                return i
            found_count += 1
    return -1


#  aggregate multiple paths based on their confidence scores.
def aggregate_paths_based_on_scores(paths):
    answer_scores = {}
    best_full_ans = None
    for answer, delta, final_answer in paths:
        answer_scores[final_answer] = answer_scores.get(
            final_answer, 0) + delta

    best_answer = max(answer_scores, key=answer_scores.get)

    for answer, delta, final_answer in paths:
        if final_answer == best_answer:
            best_full_ans = answer
            break

    return best_full_ans, answer_scores[best_answer], best_answer
