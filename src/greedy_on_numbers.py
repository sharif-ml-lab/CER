import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.uncertainty import aggregate_paths_based_on_scores, _handle_all_decoding, _handle_last_decoding
from src.sampling import sampling_strategy


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
        nlp,
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
            finished = torch.zeros(
                generated_ids.shape[0], dtype=torch.bool, device=device)

            # Start generation loop
            for step in range(max_new_tokens):
                # If all sequences are finished, break early
                if finished.all():
                    break

                outputs = model(generated_ids, attention_mask=gen_masks)
                # [batch_size*k, seq_len + step, vocab_size]
                logits = outputs.logits
                # [batch_size*k, vocab_size]
                next_token_logits = logits[:, -1, :]

                collected_logits.append(
                    next_token_logits.detach().cpu().clone())

                # Find the highest-probability token across the batch
                top_token_ids = torch.argmax(
                    next_token_logits, dim=-1)  # [batch_size*k]
                top_token_strs = tokenizer.batch_decode(
                    top_token_ids.unsqueeze(-1))

                # Prepare a placeholder for the next tokens
                next_token_ids_final = torch.zeros_like(
                    top_token_ids, dtype=torch.long, device=device)

                # Iterate over each sample in the expanded batch to apply the mixed decoding logic
                for i in range(top_token_ids.shape[0]):
                    # Skip if this sample is already finished
                    if finished[i]:
                        next_token_ids_final[i] = tokenizer.eos_token_id
                        continue

                    # sampling strategy: (1) "greedy_number_sampling", (2) "entropy_sampling", (3) confidence_sampling
                    chosen_token_id = sampling_strategy(tokenizer=tokenizer, top_token_strs=top_token_strs,
                                                        top_token_ids=top_token_ids, next_token_logits=next_token_logits, generated_ids=generated_ids, output_scores=torch.stack(
                                                            [step_logits[i] for step_logits in collected_logits], dim=0), original_length=tokenized_batch["input_ids"][i].shape[0], batch_sample_id=i, temperature=temperature, sampling_strategy="confidence_sampling")

                   # If chosen token is EOS, mark as finished
                    if chosen_token_id == tokenizer.eos_token_id:
                        finished[i] = True

                    next_token_ids_final[i] = chosen_token_id

                # Append the chosen token to sequences in the batch
                # [batch_size*k, 1]
                next_token_ids_final = next_token_ids_final.unsqueeze(-1)
                generated_ids = torch.cat(
                    [generated_ids, next_token_ids_final], dim=1)

                # Update the attention mask
                step_mask = torch.ones(
                    (gen_masks.shape[0], 1), dtype=torch.long, device=device)
                gen_masks = torch.cat([gen_masks, step_mask], dim=-1)

            for i in range(batch_size):

                # The original sequence length was input_ids[original_i].shape[0] for the branch token
                original_length = tokenized_batch["input_ids"][i].shape[0]
                # Extract the generated portion (beyond the original + branch token)
                answer_ids = generated_ids[i, original_length:]
                answer_text = tokenizer.decode(
                    answer_ids, skip_special_tokens=True)

                # Gather the per-step logits for this sequence
                # Each step in collected_logits has shape [batch_size*k, vocab_size],
                # so we index [step][idx].
                output_scores = torch.stack(
                    [step_logits[i] for step_logits in collected_logits], dim=0)

                # Decide which decoding function to apply
                if decoding_mode == "last":
                    result = _handle_last_decoding(
                        tokenizer, device, answer_text, output_scores, answer_ids, confidence_method, multihop, doc=None
                    )
                else:
                    result = _handle_all_decoding(tokenizer, device, answer_text, output_scores, answer_ids, scoring_mode,
                                                  confidence_method, multihop, doc=None, random_selection=False, random_selection_number_words=None)

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
        nlp,
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
            branch_ids = torch.cat(
                [input_ids[i], token_idx.unsqueeze(0)], dim=0)
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
    expanded_input_ids = torch.stack(all_start_ids, dim=0).to(
        device)  # [batch_size*k, seq_len+1]
    expanded_attention_masks = torch.stack(all_start_masks, dim=0).to(
        device)  # [batch_size*k, seq_len+1]

    # Initialize generation tracking tensors
    # We'll treat expanded_input_ids as the starting sequence for generation
    generated_ids = expanded_input_ids
    gen_masks = expanded_attention_masks

    # Collect logits for each generation step:
    # collected_logits[step] will be shape [batch_size*k, vocab_size]
    collected_logits = []

    # Track finished states for each sequence
    finished = torch.zeros(
        generated_ids.shape[0], dtype=torch.bool, device=device)

    # Start generation loop
    for step in range(max_new_tokens):
        # If all sequences are finished, break early
        if finished.all():
            break

        with torch.no_grad():
            outputs = model(generated_ids, attention_mask=gen_masks)
            # [batch_size*k, seq_len + step, vocab_size]
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]  # [batch_size*k, vocab_size]

        collected_logits.append(next_token_logits.detach().cpu())

        # Find the highest-probability token across the batch
        top_token_ids = torch.argmax(
            next_token_logits, dim=-1)  # [batch_size*k]
        top_token_strs = tokenizer.batch_decode(top_token_ids.unsqueeze(-1))

        # Prepare a placeholder for the next tokens
        next_token_ids_final = torch.zeros_like(
            top_token_ids, dtype=torch.long, device=device)

        # Iterate over each sample in the expanded batch to apply the mixed decoding logic
        for i in range(top_token_ids.shape[0]):
            # Skip if this sample is already finished
            if finished[i]:
                next_token_ids_final[i] = tokenizer.eos_token_id
                continue

            # sampling strategy: (1) "greedy_number_sampling", (2) "entropy_sampling", (3) confidence_sampling
            chosen_token_id = sampling_strategy(tokenizer=tokenizer, top_token_strs=top_token_strs,
                                                top_token_ids=top_token_ids, next_token_logits=next_token_logits, generated_ids=generated_ids, output_scores=torch.stack(
                                                    [step_logits[i] for step_logits in collected_logits], dim=0), original_length=tokenized_batch["input_ids"][i].shape[0], batch_sample_id=i, temperature=temperature, sampling_strategy="confidence_sampling")

            # If chosen token is EOS, mark as finished
            if chosen_token_id == tokenizer.eos_token_id:
                finished[i] = True

            next_token_ids_final[i] = chosen_token_id

        # Append the chosen token to sequences in the batch
        # [batch_size*k, 1]
        next_token_ids_final = next_token_ids_final.unsqueeze(-1)
        generated_ids = torch.cat([generated_ids, next_token_ids_final], dim=1)

        # Update the attention mask
        step_mask = torch.ones(
            (gen_masks.shape[0], 1), dtype=torch.long, device=device)
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
        output_scores = torch.stack([step_logits[idx]
                                    for step_logits in collected_logits], dim=0)

        # Decide which decoding function to apply
        if decoding_mode == "last":
            result = _handle_last_decoding(
                tokenizer, device, answer_text, output_scores, answer_ids, confidence_method, multihop, doc=None
            )
        else:
            result = _handle_all_decoding(tokenizer, device, answer_text, output_scores, answer_ids, scoring_mode,
                                          confidence_method, multihop, doc=None, random_selection=False, random_selection_number_words=None)

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
        nlp,
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

    # (batch_size)
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

    # {"input_ids": (batch_size, seq_len), "attention_mask": (batch_size, seq_len)}
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
            nlp=nlp,
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
            nlp=nlp,
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
