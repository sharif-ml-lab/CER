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
        step_decomposition,
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

        # print(batch_answer_texts)
        # print('-----')
        # exit()

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
                                              confidence_method, multihop, doc, step_decomposition, nlp)

            # Only append valid results
            if result is not None:
                paths[i].append(result)

    return paths


def cer_decode(
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
        step_decomposition,
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

    if baseline_cot == "k-seperate":
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
            step_decomposition=step_decomposition,
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
