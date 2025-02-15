import torch
import gc

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.uncertainty import extract_last_numerical_value, extract_final_answer, extract_proper_nouns, aggregate_paths_based_on_scores
from src.utils import construct_prompt, postprocess_final_answer


def _k_generation(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_questions,
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
        do_sample,
        multihop,
        nlp,
):

    batch_size = len(batch_questions)
    paths = [[] for _ in range(batch_size)]
    batch_answers = []
    batch_final_answers = []

    for _ in range(k):
        torch.cuda.empty_cache()
        gc.collect()

        batch_answers = []
        batch_final_answers = []
        tokenized_batch = batch_messages_creation(
            tokenizer, batch_questions, batch_answers, multihop, device)

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

        batch_answer_texts = []
        for i in range(batch_size):
            generated_sequence = batch_output.sequences[i]
            input_length = tokenized_batch["input_ids"][i].shape[0]

            answer_ids = generated_sequence[input_length:]
            answer_text = tokenizer.decode(
                answer_ids, skip_special_tokens=True)
            batch_answer_texts.append(answer_text)

        if multihop:
            batch_docs = list(nlp.pipe(batch_answer_texts))

        for i in range(batch_size):
            answer_text = batch_answer_texts[i]
            doc = batch_docs[i] if multihop else None
            batch_answers.append(answer_text)

            if not multihop:
                final_answer = postprocess_final_answer(
                    extract_last_numerical_value(answer_text))
            else:
                all_values = extract_proper_nouns(doc)
                final_answer = extract_final_answer(answer_text)

                if not final_answer and all_values:
                    final_answer = final_answer if final_answer else all_values[-1]

            batch_final_answers.append(final_answer)

        for i in range(batch_size):
            answer_text = batch_answers[i]
            final_answer = batch_final_answers[i]
            paths[i].append(
                (answer_text, 1.0, final_answer))

    return paths


def batch_messages_creation(tokenizer, batch_questions, batch_answers, multihop, device):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    batch_messages = []
    for question in batch_questions:
        batch_messages.append([{"role": "user", "content": construct_prompt(
            question=question,
            multihop=multihop,
            use_base_prompt=True)}])

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

    tokenized_batch = tokenized_batch.to(device)
    return tokenized_batch


def greedy_baseline(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_questions,
        aggregate_paths,
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
    do_sample = False

    paths_for_batch = _k_generation(
        model=model,
        tokenizer=tokenizer,
        batch_questions=batch_questions,
        device=device,
        k=1,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=early_stopping,
        do_sample=do_sample,
        multihop=multihop,
        nlp=nlp,
    )

    # print(batch_questions)
    # print(40*'-')

    # If no paths returned, ensure we have a default result for each input.
    if not paths_for_batch or len(paths_for_batch) != len(batch_questions):
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
