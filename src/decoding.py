import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.uncertainty import aggregate_paths_based_on_scores, _handle_all_decoding, _handle_last_decoding


# model.generate k times, each time generating a single path.
def _k_seperate_generation(
        tokenized_batch,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device,
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
        confidence_method,
        multihop,
        nlp,
        random_selection,
        random_selection_number_words,
):
    # Prepare a list of lists to store paths for each item in the batch
    batch_size = tokenized_batch["input_ids"].shape[0]
    paths = [[] for _ in range(batch_size)]

    for _ in range(k):
        # Generate results for the entire batch at once
        batch_output = model.generate(
            **tokenized_batch,
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

        batch_output_scores = []
        batch_answer_texts = []
        batch_answer_ids = []
        for i in range(batch_size):
            # Retrieve the generated sequence for this batch element
            generated_sequence = batch_output.sequences[i]
            input_length = tokenized_batch["input_ids"][i].shape[0]

            answer_ids = generated_sequence[input_length:]
            answer_text = tokenizer.decode(
                answer_ids, skip_special_tokens=True)
            output_scores = torch.stack([x[i] for x in batch_output.scores])

            # Save the result of the batch
            batch_answer_ids.append(answer_ids)
            batch_answer_texts.append(answer_text)
            batch_output_scores.append(output_scores)

        if multihop:
            batch_docs = list(nlp.pipe(batch_answer_texts))

        for i in range(batch_size):
            answer_text = batch_answer_texts[i]
            answer_ids = batch_answer_ids[i]
            output_scores = batch_output_scores[i]
            doc = batch_docs[i] if multihop else None

            if decoding_mode == "last":
                result = _handle_last_decoding(tokenizer, device, answer_text, output_scores, answer_ids,
                                               confidence_method, multihop, doc, )
            else:
                result = _handle_all_decoding(tokenizer, device, answer_text, output_scores, answer_ids, scoring_mode,
                                              confidence_method, multihop, doc,
                                              random_selection,
                                              random_selection_number_words, )

            # Only append valid results
            if result is not None:
                paths[i].append(result)

    # for testing
    # for path in paths:
    #     print(path)
    # print("======================================")

    return paths


#  we first pick the top-k next tokens and then generate one path per token.
def _k_branch_generation(
        tokenized_batch,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device,
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
        confidence_method,
        multihop,
        nlp,
        random_selection,
        random_selection_number_words,
):
    input_ids = tokenized_batch["input_ids"]
    attention_mask = tokenized_batch["attention_mask"]
    batch_size = input_ids.shape[0]

    # We'll accumulate all top-k branch starts into these lists
    all_start_ids = []
    all_start_masks = []
    index_map = []  # Will store (original_batch_idx) for each expanded example

    paths = []
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.logits has shape [batch_size, seq_len, vocab_size]

    # For each item in the batch, pick top-k tokens from the last position
    for i in range(batch_size):
        # Shape [vocab_size]
        final_token_logits = outputs.logits[i, -1, :]
        _, top_k_indices = torch.topk(final_token_logits, k)

        for token_idx in top_k_indices:
            # Extend the input_ids with a single top-k token
            branch_ids = torch.cat(
                [input_ids[i], token_idx.unsqueeze(0)], dim=0)
            # Extend attention_mask with a single 1
            branch_mask = torch.cat(
                [attention_mask[i], torch.ones(
                    (1), dtype=torch.long, device=device)],
                dim=0
            )
            all_start_ids.append(branch_ids)
            all_start_masks.append(branch_mask)
            index_map.append(i)

    # Stack all branches into a single expanded batch
    expanded_input_ids = torch.stack(all_start_ids, dim=0).to(device)
    expanded_attention_masks = torch.stack(all_start_masks, dim=0).to(device)

    # Generate for the entire expanded batch in one pass
    expanded_output = model.generate(
        input_ids=expanded_input_ids,
        attention_mask=expanded_attention_masks,
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

    # Prepare a list of lists to store generated paths for each item
    paths = [[] for _ in range(batch_size)]

    # Process each expanded result and map it back to the original batch item
    batch_output_scores = []
    batch_answer_texts = []
    batch_answer_ids = []
    batch_originals = []

    for idx, generated_sequence in enumerate(expanded_output.sequences):
        # Determine which original item this belongs to
        original_i = index_map[idx]
        # The original sequence length was input_ids[i].shape[0] + 1 for the branch token
        original_length_plus_one = input_ids[original_i].shape[0] + 1
        answer_ids = generated_sequence[original_length_plus_one:]
        answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)

        # Scores for the entire sequence; typically a list of step logits
        # We'll index them if needed in the decoding functions
        output_scores = torch.stack([x[idx] for x in expanded_output.scores])

        batch_answer_texts.append(answer_text)
        batch_answer_ids.append(answer_ids)
        batch_output_scores.append(output_scores)
        batch_originals.append(original_i)

    if multihop:
        batch_docs = list(nlp.pipe(batch_answer_texts))

    for i, answer_text in enumerate(batch_answer_texts):
        original_i = batch_originals[i]
        doc = batch_docs[i] if multihop else None
        answer_ids = batch_answer_ids[i]
        output_scores = batch_output_scores[i]

        # Decide which decoding function to apply
        if decoding_mode == "last":
            result = _handle_last_decoding(
                tokenizer, device, answer_text, output_scores, answer_ids, confidence_method, multihop, doc, )
        else:
            result = _handle_all_decoding(
                tokenizer, device, answer_text, output_scores, answer_ids, scoring_mode, confidence_method, multihop,
                doc,
                random_selection,
                random_selection_number_words,
            )

        # Append valid result to the corresponding batch item
        if result is not None:
            paths[original_i].append(result)

    return paths


# cot-decoding as originally implemented.


def cot_decode(
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
        random_selection,
        random_selection_number_words,
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
    if baseline_cot == "k-branch":
        # _k_branch_generation should return a list of paths for each example
        # Each element is a list of (generated_text, confidence_score, final_answer).
        paths_for_batch = _k_branch_generation(
            tokenized_batch=tokenized_batch,
            model=model,
            tokenizer=tokenizer,
            device=device,
            k=k,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            decoding_mode=decoding_mode,
            scoring_mode=scoring_mode,
            do_sample=do_sample,
            confidence_method=confidence_method,
            multihop=multihop,
            nlp=nlp,
            random_selection=random_selection,
            random_selection_number_words=random_selection_number_words,
        )

    elif baseline_cot == "k-seperate":
        # _k_seperate_generation should return a list of paths for each example
        # Each element is a list of (generated_text, confidence_score, final_answer).
        paths_for_batch = _k_seperate_generation(
            tokenized_batch=tokenized_batch,
            model=model,
            tokenizer=tokenizer,
            device=device,
            k=k,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            decoding_mode=decoding_mode,
            scoring_mode=scoring_mode,
            do_sample=do_sample,
            confidence_method=confidence_method,
            multihop=multihop,
            nlp=nlp,
            random_selection=random_selection,
            random_selection_number_words=random_selection_number_words,
        )
    else:
        raise ValueError(f"Unsupported baseline_cot mode: {baseline_cot}")

    # for testing
    # print(paths_for_batch)

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
