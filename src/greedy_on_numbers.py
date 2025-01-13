import torch
import spacy
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np

from src.utils import extract_all_numerical_values, extract_last_numerical_value, extract_proper_nouns, \
    extract_final_answer, postprocess_final_answer
from src.uncertainty import _find_subsequence_indices, calculate_confidence_for_final_answer, \
    aggregate_paths_based_on_scores


# extract the final numerical value.


def _handle_last_decoding(
        tokenizer: PreTrainedTokenizer,
        device,
        answer_text,
        output_scores,
        answer_ids,
        confidence_method,
        multihop, ):
    if not multihop:
        final_answer = extract_last_numerical_value(answer_text)
    else:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(answer_text)
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
    confidence = calculate_confidence_for_final_answer(final_answer_scores,
                                                       torch.tensor(final_answer_ids, device=device), confidence_method)

    if not multihop:
        final_answer = postprocess_final_answer(final_answer)

    return answer_text, confidence, final_answer


# extract all numerical values.
def _handle_all_decoding(
        tokenizer: PreTrainedTokenizer,
        device,
        answer_text,
        output_scores,
        answer_ids,
        scoring_mode,
        confidence_method,
        multihop, ):
    if not multihop:
        all_values = extract_all_numerical_values(answer_text)
    else:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(answer_text)
        all_values = extract_proper_nouns(doc)
        final_answer = extract_final_answer(answer_text)

        if not all_values and final_answer:
            all_values.append(final_answer)

        elif all_values[-1] != final_answer and final_answer:
            all_values.append(final_answer)

        # for testing
        # print(answer_text)
        # print(all_values)

    if not all_values:
        return None

    answer_ids_list = answer_ids.tolist()
    confidence_sum = 0.0
    min_conf = 10
    max_conf = -1
    total_valid_values = 0
    seen_dict = {}

    for num_idx, num_value in enumerate(all_values):
        seen_dict[num_value] = seen_dict.get(num_value, 0) + 1
        num_value_ids = tokenizer.encode(num_value, add_special_tokens=False)
        occurrence_count = seen_dict[num_value]

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
        conf_val = calculate_confidence_for_final_answer(num_value_scores, torch.tensor(num_value_ids, device=device),
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
        else:
            raise NotImplementedError

        if not multihop:
            final_answer = postprocess_final_answer(all_values[-1])
        else:
            final_answer = all_values[-1]

            # for testing
            # print(final_answer)

        return answer_text, confidence, final_answer

    return None


# model.generate k times, each time generating a single path.
def _k_seperate_generation(
        tokenized_batch,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device,
        k,
        max_new_tokens,
        temperature,
        decoding_mode,
        scoring_mode,
        confidence_method,
        multihop,
):
    # Prepare a list of lists to store paths for each item in the batch
    batch_size = tokenized_batch["input_ids"].shape[0]
    paths = [[] for _ in range(batch_size)]

    with torch.no_grad():
        for _ in range(k):
            # Initialize generation tracking tensors
            # We'll treat expanded_input_ids as the starting sequence for generation
            generated_ids = tokenized_batch["input_ids"]
            gen_masks = tokenized_batch["attention_mask"]

            # Collect logits for each generation step:
            # collected_logits[step] will be shape [batch_size*k, vocab_size]
            collected_logits = []

            # Track finished states for each sequence
            finished = torch.zeros(generated_ids.shape[0], dtype=torch.bool, device=device)

            # Start generation loop
            for step in range(max_new_tokens):
                # If all sequences are finished, break early
                if finished.all():
                    break

                with torch.no_grad():
                    outputs = model(generated_ids, attention_mask=gen_masks)
                    logits = outputs.logits  # [batch_size*k, seq_len + step, vocab_size]
                    next_token_logits = logits[:, -1, :]  # [batch_size*k, vocab_size]

                collected_logits.append(next_token_logits.detach().cpu().clone())

                # Find the highest-probability token across the batch
                top_token_ids = torch.argmax(next_token_logits, dim=-1)  # [batch_size*k]
                top_token_strs = tokenizer.batch_decode(top_token_ids.unsqueeze(-1))

                # Prepare a placeholder for the next tokens
                next_token_ids_final = torch.zeros_like(top_token_ids, dtype=torch.long, device=device)

                # Iterate over each sample in the expanded batch to apply the mixed decoding logic
                for i in range(top_token_ids.shape[0]):
                    # Skip if this sample is already finished
                    if finished[i]:
                        next_token_ids_final[i] = tokenizer.eos_token_id
                        continue

                    # Check if the highest-probability token is numeric
                    if top_token_strs[i] == tokenizer.eos_token_id:
                        chosen_token_id = tokenizer.eos_token_id
                    elif top_token_strs[i].strip().isnumeric():
                        chosen_token_id = top_token_ids[i]
                    else:
                        # Otherwise, use temperature sampling
                        adjusted_logits = next_token_logits[i] / temperature
                        adjusted_prob = torch.softmax(adjusted_logits, dim=-1)
                        sampled_id = torch.multinomial(adjusted_prob, num_samples=1)
                        chosen_token_id = sampled_id.item()

                    # If chosen token is EOS, mark as finished
                    if chosen_token_id == tokenizer.eos_token_id:
                        finished[i] = True

                    next_token_ids_final[i] = chosen_token_id

                # Append the chosen token to sequences in the batch
                next_token_ids_final = next_token_ids_final.unsqueeze(-1)  # [batch_size*k, 1]
                generated_ids = torch.cat([generated_ids, next_token_ids_final], dim=1)

                # Update the attention mask
                step_mask = torch.ones((gen_masks.shape[0], 1), dtype=torch.long, device=device)
                gen_masks = torch.cat([gen_masks, step_mask], dim=-1)

            for i in range(batch_size):
                # The original sequence length was input_ids[original_i].shape[0] for the branch token
                original_length = tokenized_batch["input_ids"][i].shape[0]
                # Extract the generated portion (beyond the original + branch token)
                answer_ids = generated_ids[i, original_length:]
                answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)

                # Gather the per-step logits for this sequence
                # Each step in collected_logits has shape [batch_size*k, vocab_size],
                # so we index [step][idx].
                output_scores = torch.stack([step_logits[i] for step_logits in collected_logits], dim=0)

                # Decide which decoding function to apply
                if decoding_mode == "last":
                    result = _handle_last_decoding(
                        tokenizer, device, answer_text, output_scores, answer_ids, confidence_method, multihop
                    )
                else:
                    result = _handle_all_decoding(
                        tokenizer, device, answer_text, output_scores, answer_ids, scoring_mode, confidence_method,
                        multihop
                    )

                # Only append valid results
                if result is not None:
                    paths[i].append(result)

    return paths


#  we first pick the top-k next tokens and then generate one path per token.
def _k_branch_generation(
        tokenized_batch,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device,
        k,
        max_new_tokens,
        temperature,
        decoding_mode,
        scoring_mode,
        confidence_method,
        multihop,
):
    input_ids = tokenized_batch["input_ids"]
    attention_mask = tokenized_batch["attention_mask"]
    batch_size = input_ids.shape[0]

    # We'll accumulate all top-k branch starts into these lists
    all_start_ids = []
    all_start_masks = []
    index_map = []  # Will store the original batch index for each expanded example

    # Initial forward pass to get logits
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.logits has shape [batch_size, seq_len, vocab_size]

    # For each item in the batch, pick top-k tokens from the last position
    for i in range(batch_size):
        final_token_logits = outputs.logits[i, -1, :]  # [vocab_size]
        _, top_k_indices = torch.topk(final_token_logits, k)

        for token_idx in top_k_indices:
            # Extend the input_ids with a single top-k token
            branch_ids = torch.cat([input_ids[i], token_idx.unsqueeze(0)], dim=0)
            # Extend attention_mask with a single 1
            branch_mask = torch.cat(
                [
                    attention_mask[i],
                    torch.ones((1,), dtype=torch.long, device=device)
                ],
                dim=0
            )
            all_start_ids.append(branch_ids)
            all_start_masks.append(branch_mask)
            index_map.append(i)

    # Stack all branches into a single expanded batch
    expanded_input_ids = torch.stack(all_start_ids, dim=0).to(device)  # [batch_size*k, seq_len+1]
    expanded_attention_masks = torch.stack(all_start_masks, dim=0).to(device)  # [batch_size*k, seq_len+1]

    # Initialize generation tracking tensors
    # We'll treat expanded_input_ids as the starting sequence for generation
    generated_ids = expanded_input_ids
    gen_masks = expanded_attention_masks

    # Collect logits for each generation step:
    # collected_logits[step] will be shape [batch_size*k, vocab_size]
    collected_logits = []

    # Track finished states for each sequence
    finished = torch.zeros(generated_ids.shape[0], dtype=torch.bool, device=device)

    # Start generation loop
    for step in range(max_new_tokens):
        # If all sequences are finished, break early
        if finished.all():
            break

        with torch.no_grad():
            outputs = model(generated_ids, attention_mask=gen_masks)
            logits = outputs.logits  # [batch_size*k, seq_len + step, vocab_size]
            next_token_logits = logits[:, -1, :]  # [batch_size*k, vocab_size]

        collected_logits.append(next_token_logits.detach().cpu())

        # Find the highest-probability token across the batch
        top_token_ids = torch.argmax(next_token_logits, dim=-1)  # [batch_size*k]
        top_token_strs = tokenizer.batch_decode(top_token_ids.unsqueeze(-1))

        # Prepare a placeholder for the next tokens
        next_token_ids_final = torch.zeros_like(top_token_ids, dtype=torch.long, device=device)

        # Iterate over each sample in the expanded batch to apply the mixed decoding logic
        for i in range(top_token_ids.shape[0]):
            # Skip if this sample is already finished
            if finished[i]:
                next_token_ids_final[i] = tokenizer.eos_token_id
                continue

            # Check if the highest-probability token is numeric
            if top_token_strs[i] == tokenizer.eos_token_id:
                chosen_token_id = tokenizer.eos_token_id
            elif top_token_strs[i].strip().isnumeric():
                chosen_token_id = top_token_ids[i]
            else:
                # Otherwise, use temperature sampling
                adjusted_logits = next_token_logits[i] / temperature
                adjusted_prob = torch.softmax(adjusted_logits, dim=-1)
                sampled_id = torch.multinomial(adjusted_prob, num_samples=1)
                chosen_token_id = sampled_id.item()

            # If chosen token is EOS, mark as finished
            if chosen_token_id == tokenizer.eos_token_id:
                finished[i] = True

            next_token_ids_final[i] = chosen_token_id

        # Append the chosen token to sequences in the batch
        next_token_ids_final = next_token_ids_final.unsqueeze(-1)  # [batch_size*k, 1]
        generated_ids = torch.cat([generated_ids, next_token_ids_final], dim=1)

        # Update the attention mask
        step_mask = torch.ones((gen_masks.shape[0], 1), dtype=torch.long, device=device)
        gen_masks = torch.cat([gen_masks, step_mask], dim=-1)

    # At this point, 'generated_ids' contains the final token sequences
    # and 'collected_logits' holds the logits for each generation step.
    # Prepare a list of lists to store generated paths for each item
    paths = [[] for _ in range(batch_size)]

    # Process each expanded result and map it back to the original batch item
    for idx in range(generated_ids.shape[0]):
        original_i = index_map[idx]

        # The original sequence length was input_ids[original_i].shape[0] for the branch token
        original_length = input_ids[original_i].shape[0]
        # Extract the generated portion (beyond the original + branch token)
        answer_ids = generated_ids[idx, original_length:]
        answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)

        # Gather the per-step logits for this sequence
        # Each step in collected_logits has shape [batch_size*k, vocab_size],
        # so we index [step][idx].
        output_scores = torch.stack([step_logits[idx] for step_logits in collected_logits], dim=0)

        # Decide which decoding function to apply
        if decoding_mode == "last":
            result = _handle_last_decoding(
                tokenizer, device, answer_text, output_scores, answer_ids, confidence_method, multihop
            )
        else:
            result = _handle_all_decoding(
                tokenizer, device, answer_text, output_scores, answer_ids, scoring_mode, confidence_method, multihop
            )

        # Append valid result to the corresponding batch item
        if result is not None:
            paths[original_i].append(result)

    return paths


def special_greedy_decode(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_messages,
        sampling_mode,
        scoring_mode,
        k,
        aggregate_paths,
        decoding_mode,
        baseline_cot,
        confidence_method,
        multihop,
        num_beams=1,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
        length_penalty=1.0,
        no_repeat_ngram_size=0,
        max_new_tokens=1024,
        early_stopping=False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine sampling mode.
    if sampling_mode == "temperature":
        do_sample = True
    elif sampling_mode == "greedy":
        do_sample = False
    else:
        raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")

    # Ensure the tokenizer has a pad token ID.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    batch_template_messages = []
    for message in batch_messages:
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            input_text = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            input_text = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in message])
            input_text += "\nassistant:"

        batch_template_messages.append(input_text)

    tokenized_batch = tokenizer(
        batch_template_messages,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # Move the batch dict to the same device
    tokenized_batch = tokenized_batch.to(device)

    # Branch or separate generation depending on the baseline_cot mode.
    if baseline_cot == "branch_greedy_special":
        # _k_branch_generation should return a list of paths for each example
        # Each element is a list of (generated_text, confidence_score, final_answer).
        paths_for_batch = _k_branch_generation(
            tokenized_batch=tokenized_batch,
            model=model,
            tokenizer=tokenizer,
            device=device,
            k=k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            decoding_mode=decoding_mode,
            scoring_mode=scoring_mode,
            confidence_method=confidence_method,
            multihop=multihop,
        )

    elif baseline_cot == "seperated_greedy_special":
        # _k_seperate_generation should return a list of paths for each example
        # Each element is a list of (generated_text, confidence_score, final_answer).
        paths_for_batch = _k_seperate_generation(
            tokenized_batch=tokenized_batch,
            model=model,
            tokenizer=tokenizer,
            device=device,
            k=k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            decoding_mode=decoding_mode,
            scoring_mode=scoring_mode,
            confidence_method=confidence_method,
            multihop=multihop,
        )
    else:
        raise ValueError(f"Unsupported baseline_cot mode: {baseline_cot}")

    # If no paths returned, ensure we have a default result for each input.
    if not paths_for_batch or len(paths_for_batch) != len(batch_messages):
        raise RuntimeError("There was a problem with inferring this batch")

    # For each example, pick either an aggregated path or the best path by score.
    all_decoded = []
    for i, single_example_paths in enumerate(paths_for_batch):
        if not single_example_paths:
            # If no generation for a specific example, append defaults.
            all_decoded.append(("", 0.0, ""))
            continue

        if aggregate_paths:
            # Aggregate the paths if needed.
            aggregated = aggregate_paths_based_on_scores(single_example_paths)
            all_decoded.append(aggregated)
        else:
            # Otherwise pick the path with the best confidence.
            all_decoded.append(max(single_example_paths, key=lambda x: x[1]))

    return all_decoded
