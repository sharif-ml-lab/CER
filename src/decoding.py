from tqdm import tqdm

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np

from src.utils import extract_all_numerical_values, extract_last_numerical_value
from src.uncertainty import _find_subsequence_indices, calculate_confidence_for_final_answer, aggregate_paths_based_on_scores

# extract the final numerical value.
def _handle_last_decoding(
        tokenizer: PreTrainedTokenizer,
        device,
        answer_text,
        output_scores,
        answer_ids,
        confidence_method):
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
    confidence = calculate_confidence_for_final_answer(final_answer_scores, torch.tensor(final_answer_ids, device=device), confidence_method)
    return answer_text, confidence, final_answer


# extract all numerical values. 
def _handle_all_decoding(
        tokenizer: PreTrainedTokenizer,
        device,
        answer_text,
        output_scores,
        answer_ids,
        scoring_mode,
        confidence_method):
    
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
        conf_val = calculate_confidence_for_final_answer(num_value_scores, torch.tensor(num_value_ids, device=device), confidence_method)

        if scoring_mode == 'log':
            confidence_sum += np.log(1 + conf_val)
        elif scoring_mode == 'min':
            if conf_val < min_conf:
                min_conf = conf_val
                confidence_sum = conf_val
        elif scoring_mode == 'max':
            if conf_val > max_conf:
                max_conf = conf_val
                confidence_sum = conf_val
        elif scoring_mode == 'h_mean':
            confidence_sum += 1 / (1e-11 + conf_val)
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



# model.generate k times, each time generating a single path.
def _k_seperate_generation(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device,
        input_ids,
        attention_mask,
        k,
        max_new_tokens,
        num_beams,
        temperature,
        top_p,
        repetition_penalty,
        length_penalty,
        no_repeat_ngram_size,
        early_stopping,
        decoding_mode,
        scoring_mode,
        do_sample,
        confidence_method):
    
    paths = []
    for _ in tqdm(range(k)):
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=do_sample,
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

        if decoding_mode == "last":
            result = _handle_last_decoding(tokenizer, device, answer_text, output_scores, answer_ids, confidence_method)
        else:
            result = _handle_all_decoding(tokenizer, device, answer_text, output_scores, answer_ids, scoring_mode, confidence_method)

        if result is not None:
            paths.append(result)

    return paths


#  we first pick the top-k next tokens and then generate one path per token.
def _k_branch_generation(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device,
        input_ids,
        attention_mask,
        k,
        max_new_tokens,
        num_beams,
        temperature,
        top_p,
        repetition_penalty,
        length_penalty,
        no_repeat_ngram_size,
        early_stopping,
        decoding_mode,
        scoring_mode,
        do_sample,
        confidence_method):
    
    paths = []
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        first_token_logits = outputs.logits[0, -1, :]
        _, top_k_indices = torch.topk(first_token_logits, k)

    for idx in top_k_indices:
        start_ids = torch.cat([input_ids, idx.unsqueeze(0).unsqueeze(0)], dim=-1)
        start_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=-1)

        output = model.generate(
            start_ids,
            attention_mask=start_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
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

        if decoding_mode == "last": # last nuemrical values
            result = _handle_last_decoding(tokenizer, device, answer_text, output_scores, answer_ids, confidence_method)
        else: # all numerical values
            result = _handle_all_decoding(tokenizer, device, answer_text, output_scores, answer_ids, scoring_mode, confidence_method)

        if result is not None:
            paths.append(result)

    return paths

# cot-decoding as originally implemented.
def cot_decode(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        messages,
        sampling_mode,
        scoring_mode,
        k,
        aggregate_paths,
        decoding_mode,
        baseline_cot,
        confidence_method,
        num_beams=1,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
        length_penalty=1.0,
        no_repeat_ngram_size=0,
        max_new_tokens=512,
        do_sample=True,
        early_stopping= False):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


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

    if sampling_mode == "temperature": # to set parameters for temperature-based sampling (different each time)
        do_sample = True

    elif sampling_mode == "greedy": # to set parameters for greedy-based sampling (unique each time)
        do_sample = False

    if baseline_cot == "k-branch": # make k branches and then continue each one with sampling mode
        paths = _k_branch_generation(
                model, tokenizer, device, input_ids, attention_mask, k,
                max_new_tokens, num_beams, temperature, top_p,
                repetition_penalty, length_penalty, no_repeat_ngram_size,
                early_stopping, decoding_mode, scoring_mode, do_sample, confidence_method)
    
    elif baseline_cot == "k-seperate": # make k distinict paths with sampling mode
        paths = _k_seperate_generation(
            model, tokenizer, device, input_ids, attention_mask, k,
            max_new_tokens, num_beams, temperature, top_p,
            repetition_penalty, length_penalty, no_repeat_ngram_size,
            early_stopping, decoding_mode, scoring_mode, do_sample, confidence_method)

    else:
        raise ValueError("Unsupported sampling_mode")

    if not paths:
        return "", 0.0, ""

    if aggregate_paths:
        return aggregate_paths_based_on_scores(paths)
    else:
        return max(paths, key=lambda x: x[1])
