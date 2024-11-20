import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Tuple, Dict, Optional
import re
import numpy as np


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def extract_last_numerical_value(text: str) -> Optional[str]:
    """
    Extract the last numerical value from a given text.

    Args:
        text: The text from which to extract the numerical value.

    Returns:
        The last numerical value if found, otherwise None.
    """
    matches = re.findall(r'\b\d+\.?\d*\b', text)
    return matches[-1] if matches else None


def extract_all_numerical_values(text: str) -> List[str]:
    """
    Extract all numerical values from a given text.

    Args:
        text: The text from which to extract numerical values.

    Returns:
        A list of all numerical values found in the text.
    """
    return re.findall(r'\b\d+\.?\d*\b', text)


def calculate_confidence_for_final_answer(logits: List[torch.Tensor], answer_ids: torch.Tensor) -> float:
    """
    Calculate the confidence score (Δ) as specified in the paper.

    Args:
        logits: List of logits for each decoding step
        answer_ids: Tensor of token ids for the answer

    Returns:
        Confidence score (Δ)
    """
    confidence_sum = 0.0
    valid_tokens = 0
    for t, token_id in enumerate(answer_ids):
        if t >= len(logits):
            break
        token_logits = logits[t]
        probs = torch.softmax(token_logits, dim=-1)
        if probs.size(-1) > 1:
            top_2_probs, _ = torch.topk(probs, min(2, probs.size(-1)))
            if top_2_probs.size(-1) > 1:
                confidence_sum += (top_2_probs[-1][0] - top_2_probs[-1][1]).item()
            else:
                confidence_sum += 1.0  # Max confidence if there's only one token
        else:
            confidence_sum += 1.0  # Max confidence if there's only one token
        valid_tokens += 1

    return confidence_sum / valid_tokens if valid_tokens > 0 else 0.0


def aggregate_paths_based_on_scores(paths: List[Tuple[str, float, str]]) -> Tuple[str, float, str]:
    """Aggregate multiple paths based on their confidence scores."""
    answer_scores = {}
    best_full_ans = None
    best_full_ans_delta = -1
    for answer, delta, final_answer in paths:
        answer_scores[final_answer] = answer_scores.get(final_answer, 0) + delta
        if best_full_ans_delta < delta:
            best_full_ans_delta = delta
            best_full_ans = answer
    best_answer = max(answer_scores, key=answer_scores.get)
    return best_full_ans, answer_scores[best_answer], best_answer


def cot_decode(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        messages: List[Dict[str, str]],
        sampling_mode="cot",
        k: int = 10,
        num_beams: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        early_stopping: bool = False,
        aggregate_paths: bool = False,
        decoding_mode: str = "baseline",
) -> Tuple[str, float, str]:
    """
    Implement CoT-decoding for a given chat input.

    Args:
        model: The Hugging Face transformer model.
        tokenizer: The associated tokenizer.
        messages: List of chat messages in the format [{"role": "user", "content": "..."}]
        k: The number of alternative tokens to consider at the first step.
        num_beams: Number of beams for beam search.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        repetition_penalty: Repetition penalty factor.
        length_penalty: Length penalty factor.
        no_repeat_ngram_size: Size of n-grams to avoid repeating.
        early_stopping: Whether to stop generation when all beams are finished.
        aggregate_paths: Whether to aggregate multiple paths.
        decoding_mode: Mode of decoding, either "baseline" or "new".

    Returns:
        A tuple containing the best path (or aggregated result) and its confidence score.
    """
    device = get_device()
    # model.to(device)

    # Use the chat template to format the input
    if tokenizer.chat_template:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback for tokenizers without chat templates
        input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        input_text += "\nassistant:"

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # Set pad_token_id if it's not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    paths = []
    if sampling_mode == "cot":
        # Get the top-k tokens for the first decoding step
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            first_token_logits = outputs.logits[0, -1, :]
            top_k_logits, top_k_indices = torch.topk(first_token_logits, k)

        for idx in top_k_indices:
            # Generate sequence starting with the selected token
            start_ids = torch.cat([input_ids, idx.unsqueeze(0).unsqueeze(0)], dim=-1)
            start_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=-1)

            output = model.generate(
                start_ids,
                attention_mask=start_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

            generated_sequence = output.sequences[0]
            answer_ids = generated_sequence[len(input_ids[0]):]
            answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)

            if decoding_mode == "baseline":
                # Extract the final numerical answer
                final_answer = extract_last_numerical_value(answer_text)
                if final_answer is not None:
                    final_answer_ids = tokenizer.encode(final_answer, add_special_tokens=False)

                    # Find the start index of the final occurrence of the final answer in the answer_ids
                    answer_ids_list = answer_ids.tolist()
                    final_answer_ids_list = final_answer_ids

                    final_answer_start_idx = -1
                    for i in range(len(answer_ids_list) - len(final_answer_ids_list) + 1):
                        if answer_ids_list[i:i + len(final_answer_ids_list)] == final_answer_ids_list:
                            final_answer_start_idx = i - 1  # because the first generated token's score is not presented in output.scores

                    if final_answer_start_idx == -1:
                        continue

                    final_answer_scores = output.scores[
                                          final_answer_start_idx: final_answer_start_idx + len(final_answer_ids)]

                    # Calculate confidence score (Δ) for the final answer only
                    confidence = calculate_confidence_for_final_answer(final_answer_scores,
                                                                       torch.tensor(final_answer_ids, device=device))
                    paths.append((answer_text, confidence, final_answer))

            elif decoding_mode == "new":
                # Extract all numerical values
                all_numerical_values = extract_all_numerical_values(answer_text)
                if all_numerical_values:
                    confidence_sum = 0.0
                    total_valid_values = 0
                    for num_value in all_numerical_values:
                        num_value_ids = tokenizer.encode(num_value, add_special_tokens=False)

                        # Find the start index of the final occurrence of the numerical value in the answer_ids
                        answer_ids_list = answer_ids.tolist()
                        num_value_ids_list = num_value_ids

                        num_value_start_idx = -1
                        for i in range(len(answer_ids_list) - len(num_value_ids_list) + 1):
                            if answer_ids_list[i:i + len(num_value_ids_list)] == num_value_ids_list:
                                num_value_start_idx = i - 1  # because the first generated token's score is not presented in output.scores

                        if num_value_start_idx == -1:
                            continue

                        num_value_scores = output.scores[num_value_start_idx: num_value_start_idx + len(num_value_ids)]

                        # Calculate confidence score (Δ) for this numerical value
                        confidence_sum += np.log(calculate_confidence_for_final_answer(num_value_scores,
                                                                                       torch.tensor(num_value_ids,
                                                                                                    device=device)))
                        total_valid_values += 1

                    if total_valid_values > 0:
                        confidence = confidence_sum.item() / total_valid_values
                        final_answer = all_numerical_values[
                            -1]  # Consider the last numerical value as the final answer for consistency
                        paths.append((answer_text, confidence, final_answer))

    elif sampling_mode == "temp":
        for idx in range(k):
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

            generated_sequence = output.sequences[0]
            answer_ids = generated_sequence[len(input_ids[0]):]
            answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)

            if decoding_mode == "baseline":
                # Extract the final numerical answer
                final_answer = extract_last_numerical_value(answer_text)
                if final_answer is not None:
                    final_answer_ids = tokenizer.encode(final_answer, add_special_tokens=False)

                    # Find the start index of the final occurrence of the final answer in the answer_ids
                    answer_ids_list = answer_ids.tolist()
                    final_answer_ids_list = final_answer_ids

                    final_answer_start_idx = -1
                    for i in range(len(answer_ids_list) - len(final_answer_ids_list) + 1):
                        if answer_ids_list[i:i + len(final_answer_ids_list)] == final_answer_ids_list:
                            final_answer_start_idx = i - 1  # because the first generated token's score is not presented in output.scores

                    if final_answer_start_idx == -1:
                        continue

                    final_answer_scores = output.scores[
                                          final_answer_start_idx: final_answer_start_idx + len(final_answer_ids)]

                    # Calculate confidence score (Δ) for the final answer only
                    confidence = calculate_confidence_for_final_answer(final_answer_scores,
                                                                       torch.tensor(final_answer_ids, device=device))
                    paths.append((answer_text, confidence, final_answer))

            elif decoding_mode == "new":
                # Extract all numerical values
                all_numerical_values = extract_all_numerical_values(answer_text)
                if all_numerical_values:
                    confidence_sum = 0.0
                    total_valid_values = 0
                    for num_value in all_numerical_values:
                        num_value_ids = tokenizer.encode(num_value, add_special_tokens=False)

                        # Find the start index of the final occurrence of the numerical value in the answer_ids
                        answer_ids_list = answer_ids.tolist()
                        num_value_ids_list = num_value_ids

                        num_value_start_idx = -1
                        for i in range(len(answer_ids_list) - len(num_value_ids_list) + 1):
                            if answer_ids_list[i:i + len(num_value_ids_list)] == num_value_ids_list:
                                num_value_start_idx = i - 1  # because the first generated token's score is not presented in output.scores

                        if num_value_start_idx == -1:
                            continue

                        num_value_scores = output.scores[num_value_start_idx: num_value_start_idx + len(num_value_ids)]

                        # Calculate confidence score (Δ) for this numerical value
                        confidence_sum += np.log(calculate_confidence_for_final_answer(num_value_scores,
                                                                                       torch.tensor(num_value_ids,
                                                                                                    device=device)))
                        total_valid_values += 1

                    if total_valid_values > 0:
                        confidence = confidence_sum.item() / total_valid_values
                        final_answer = all_numerical_values[
                            -1]  # Consider the last numerical value as the final answer for consistency
                        paths.append((answer_text, confidence, final_answer))

    if aggregate_paths:
        return aggregate_paths_based_on_scores(paths)
    else:
        return max(paths, key=lambda x: x[1])
