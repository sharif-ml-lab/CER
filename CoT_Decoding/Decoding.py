import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Tuple, Dict, Optional
import re
import numpy as np


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def extract_last_numerical_value(text: str) -> Optional[str]:
    matches = re.findall(r'\b\d+\.?\d*\b', text)
    return matches[-1] if matches else None


def extract_all_numerical_values(text: str) -> List[str]:
    return re.findall(r'\b\d+\.?\d*\b', text)


def calculate_confidence_for_final_answer(
        logits: List[torch.Tensor],
        answer_ids: torch.Tensor,
        mode: str = "default"
) -> float:
    confidence_sum = 1.0
    valid_tokens = 0

    for t, token_id in enumerate(answer_ids):
        if t >= len(logits):
            break
        token_logits = logits[t]
        probs = torch.softmax(token_logits, dim=-1)
        ans_token_prob = probs[-1][token_id]

        if mode == "default":
            confidence_sum *= ans_token_prob.item()
        elif mode == "sum":
            confidence_sum += ans_token_prob.item()
        elif mode == "entropy":
            probs = torch.clamp(probs, min=1e-12)
            entropy = - (probs * torch.log(probs)).sum(dim=-1)
            confidence_sum = 1 - entropy.item()
        elif mode == "top_2_diff_weighted":
            top_2_probs, _ = torch.topk(probs, min(2, probs.size(-1)))
            if top_2_probs.size(-1) > 1:
                confidence_sum += (top_2_probs[-1][0] - top_2_probs[-1][1]).item() * ans_token_prob.item()
            else:
                confidence_sum += 1.0
        elif mode == "top_2_diff":
            top_2_probs, _ = torch.topk(probs, min(2, probs.size(-1)))
            if top_2_probs.size(-1) > 1:
                confidence_sum += (top_2_probs[-1][0] - top_2_probs[-1][1]).item()
            else:
                confidence_sum += 1.0
        else:
            raise NotImplementedError("Unsupported confidence calculation mode")

        valid_tokens += 1

    return confidence_sum / valid_tokens if valid_tokens > 0 else 0.0


def aggregate_paths_based_on_scores(
        paths: List[Tuple[str, float, str]]
) -> Tuple[str, float, str]:
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


def _find_subsequence_indices(
        sequence: List[int],
        subsequence: List[int],
        occurrence_count: int = 1
) -> int:
    found_count = 0
    seq_len = len(sequence)
    sub_len = len(subsequence)

    for i in range(seq_len - sub_len + 1):
        if sequence[i:i + sub_len] == subsequence:
            if found_count == occurrence_count - 1:
                return i
            found_count += 1
    return -1


def _compute_confidence_for_value(
        output_scores: List[torch.Tensor],
        answer_ids: List[int],
        value_ids: List[int],
        device: torch.device,
        confidence_mode: str
) -> Optional[float]:
    value_start_idx = _find_subsequence_indices(answer_ids, value_ids, 1)
    if value_start_idx == -1:
        return None

    value_start_idx -= 1
    if value_start_idx < 0 or value_start_idx + len(value_ids) > len(output_scores):
        return None

    value_scores = output_scores[value_start_idx: value_start_idx + len(value_ids)]
    return calculate_confidence_for_final_answer(value_scores, torch.tensor(value_ids, device=device),
                                                 mode=confidence_mode)


def _handle_baseline_decoding(
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        answer_text: str,
        output_scores: List[torch.Tensor],
        answer_ids: torch.Tensor,
        confidence_mode: str
) -> Optional[Tuple[str, float, str]]:
    final_answer = extract_last_numerical_value(answer_text)
    if final_answer is None:
        return None

    final_answer_ids = tokenizer.encode(final_answer, add_special_tokens=False)
    answer_ids_list = answer_ids.tolist()

    final_answer_start_idx = _find_subsequence_indices(answer_ids_list, final_answer_ids, 1)
    if final_answer_start_idx == -1:
        return None
    final_answer_start_idx -= 1

    if final_answer_start_idx < 0 or final_answer_start_idx + len(final_answer_ids) > len(output_scores):
        return None

    final_answer_scores = output_scores[final_answer_start_idx: final_answer_start_idx + len(final_answer_ids)]
    confidence = calculate_confidence_for_final_answer(
        final_answer_scores,
        torch.tensor(final_answer_ids, device=device),
        mode=confidence_mode
    )
    return answer_text, confidence, final_answer


def _handle_new_decoding_cot_mode(
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        answer_text: str,
        output_scores: List[torch.Tensor],
        answer_ids: torch.Tensor,
        confidence_mode: str
) -> Optional[Tuple[str, float, str]]:
    all_numerical_values = extract_all_numerical_values(answer_text)
    if not all_numerical_values:
        return None

    confidence_sum = 0.0
    total_valid_values = 0
    answer_ids_list = answer_ids.tolist()

    for num_value in all_numerical_values:
        num_value_ids = tokenizer.encode(num_value, add_special_tokens=False)
        conf_val = _compute_confidence_for_value(output_scores, answer_ids_list, num_value_ids, device, confidence_mode)
        if conf_val is None:
            continue
        confidence_sum += np.log(conf_val)
        total_valid_values += 1

    if total_valid_values > 0:
        confidence = confidence_sum.item() / total_valid_values
        final_answer = all_numerical_values[-1]
        return answer_text, confidence, final_answer
    return None


def _handle_new_decoding_temp_mode(
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        answer_text: str,
        output_scores: List[torch.Tensor],
        answer_ids: torch.Tensor,
        scoring_mode: str,
        confidence_mode: str
) -> Optional[Tuple[str, float, str]]:
    all_numerical_values = extract_all_numerical_values(answer_text)
    if not all_numerical_values:
        return None

    answer_ids_list = answer_ids.tolist()
    confidence_sum = 0.0
    min_conf = 10
    max_conf = -1
    total_valid_values = 0
    seen_dict = {}

    for num_value in all_numerical_values:
        seen_dict[num_value] = seen_dict.get(num_value, 0) + 1
        num_value_ids = tokenizer.encode(num_value, add_special_tokens=False)
        occurrence_count = seen_dict[num_value]

        value_start_idx = _find_subsequence_indices(answer_ids_list, num_value_ids, occurrence_count)
        if value_start_idx == -1:
            continue

        if value_start_idx < 0 or value_start_idx + len(num_value_ids) > len(output_scores):
            continue

        num_value_scores = output_scores[value_start_idx: value_start_idx + len(num_value_ids)]
        conf_val = calculate_confidence_for_final_answer(
            num_value_scores,
            torch.tensor(num_value_ids, device=device),
            mode=confidence_mode
        )
        current_conf = np.log(1 + conf_val)

        if scoring_mode == 'log':
            confidence_sum += current_conf
        elif scoring_mode == 'min':
            if current_conf < min_conf:
                min_conf = current_conf
                confidence_sum = current_conf
        elif scoring_mode == 'max':
            if current_conf > max_conf:
                max_conf = current_conf
                confidence_sum = current_conf
        elif scoring_mode == 'h_mean':
            confidence_sum += 1 / (1e-11 + current_conf)
        else:
            raise NotImplementedError("Unsupported scoring_mode")

        total_valid_values += 1

    if total_valid_values > 0:
        if scoring_mode == 'log':
            confidence = confidence_sum.item() / total_valid_values
        elif scoring_mode in ['min', 'max']:
            confidence = confidence_sum
        elif scoring_mode == 'h_mean':
            confidence = total_valid_values / confidence_sum.item()
        else:
            raise NotImplementedError

        final_answer = all_numerical_values[-1]
        return answer_text, confidence, final_answer
    return None


def _sample_cot_paths(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        k: int,
        max_new_tokens: int,
        num_beams: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        length_penalty: float,
        no_repeat_ngram_size: int,
        early_stopping: bool,
        decoding_mode: str,
        confidence_mode: str
) -> List[Tuple[str, float, str]]:
    paths = []
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        first_token_logits = outputs.logits[0, -1, :]
        _, top_k_indices = torch.topk(first_token_logits, k)

    for idx in top_k_indices:
        start_ids = torch.cat([input_ids, idx.unsqueeze(0).unsqueeze(0)], dim=-1)
        start_mask = torch.cat([attention_mask,
                                torch.ones((1, 1), dtype=torch.long, device=device)],
                               dim=-1)

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
        output_scores = output.scores

        if decoding_mode == "baseline":
            result = _handle_baseline_decoding(tokenizer, device, answer_text, output_scores, answer_ids,
                                               confidence_mode)
        else:
            result = _handle_new_decoding_cot_mode(tokenizer, device, answer_text, output_scores, answer_ids,
                                                   confidence_mode)

        if result is not None:
            paths.append(result)

    return paths


def _sample_temp_paths(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        k: int,
        max_new_tokens: int,
        num_beams: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        length_penalty: float,
        no_repeat_ngram_size: int,
        early_stopping: bool,
        decoding_mode: str,
        scoring_mode: str,
        confidence_mode: str
) -> List[Tuple[str, float, str]]:
    paths = []
    for _ in range(k):
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
        output_scores = output.scores

        if decoding_mode == "baseline":
            result = _handle_baseline_decoding(tokenizer, device, answer_text, output_scores, answer_ids,
                                               confidence_mode)
        else:
            result = _handle_new_decoding_temp_mode(tokenizer, device, answer_text, output_scores, answer_ids,
                                                    scoring_mode, confidence_mode)

        if result is not None:
            paths.append(result)

    return paths


def cot_decode(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        messages: List[Dict[str, str]],
        sampling_mode: str,
        scoring_mode: str,
        k: int,
        num_beams: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        length_penalty: float,
        no_repeat_ngram_size: int,
        early_stopping: bool,
        aggregate_paths: bool,
        decoding_mode: str,
        confidence_mode: str
) -> Tuple[str, float, str]:
    device = get_device()

    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        input_text += "\nassistant:"

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if sampling_mode == "cot":
        paths = _sample_cot_paths(
            model, tokenizer, device, input_ids, attention_mask, k,
            max_new_tokens, num_beams, temperature, top_p,
            repetition_penalty, length_penalty, no_repeat_ngram_size,
            early_stopping, decoding_mode, confidence_mode
        )
    elif sampling_mode == "temp":
        paths = _sample_temp_paths(
            model, tokenizer, device, input_ids, attention_mask, k,
            max_new_tokens, num_beams, temperature, top_p,
            repetition_penalty, length_penalty, no_repeat_ngram_size,
            early_stopping, decoding_mode, scoring_mode, confidence_mode
        )
    else:
        raise ValueError("Unsupported sampling_mode")

    if not paths:
        return "", 0.0, ""

    if aggregate_paths:
        return aggregate_paths_based_on_scores(paths)
    else:
        return max(paths, key=lambda x: x[1])
