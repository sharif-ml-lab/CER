import torch
import gc

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.uncertainty import extract_last_numerical_value, extract_final_answer, extract_proper_nouns
from src.utils import construct_prompt, postprocess_final_answer


def get_normalized_loglikelihoods(generated_logits, generated_ids, pad_token_id):
    log_probs = F.log_softmax(generated_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=generated_ids.unsqueeze(-1)).squeeze(-1)

    if pad_token_id is not None:
        valid_mask = generated_ids != pad_token_id
        token_log_probs = token_log_probs[valid_mask]

    nll = - (token_log_probs).sum() / \
        len(token_log_probs)  # more -> high uncertainty
    return nll.item()


def aggregate_paths_based_on_scores(paths):
    # answer_scores = {}
    # best_full_ans = None
    # for answer, delta, final_answer in paths:
    #     answer_scores[final_answer] = answer_scores.get(
    #         final_answer, 0) + delta

    # best_answer = min(answer_scores, key=answer_scores.get)

    # for answer, delta, final_answer in paths:
    #     if final_answer == best_answer:
    #         best_full_ans = answer
    #         break

    # return best_full_ans, answer_scores[best_answer], best_answer

    best_answer, min_delta, best_final_answer = min(paths, key=lambda x: x[1])
    return best_answer, min_delta, best_final_answer


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
        few_shot,
        few_shot_path,
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
            tokenizer, batch_questions, batch_answers, few_shot, few_shot_path, multihop, device)

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
        batch_nlls = []
        for i in range(batch_size):
            generated_sequence = batch_output.sequences[i]
            input_length = tokenized_batch["input_ids"][i].shape[0]

            answer_ids = generated_sequence[input_length:]
            answer_text = tokenizer.decode(
                answer_ids, skip_special_tokens=True)
            output_scores = torch.stack(
                [x[i] for x in batch_output.scores])

            nll = get_normalized_loglikelihoods(
                output_scores, answer_ids, pad_token_id=tokenizer.pad_token_id)

            batch_answer_ids.append(answer_ids)
            batch_answer_texts.append(answer_text)
            batch_output_scores.append(output_scores)
            batch_nlls.append(nll)

        if multihop:
            batch_docs = list(nlp.pipe(batch_answer_texts))

        for i in range(batch_size):
            answer_text = batch_answer_texts[i]
            answer_ids = batch_answer_ids[i]
            output_scores = batch_output_scores[i]
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
            nll = batch_nlls[i]
            paths[i].append(
                (answer_text, nll, final_answer))

    return paths


def batch_messages_creation(tokenizer, batch_questions, batch_answers, few_shot, few_shot_path, multihop, device):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    batch_messages = []
    for question in batch_questions:
        batch_messages.append([{"role": "user", "content": construct_prompt(
            question=question,
            few_shot=few_shot,
            few_shot_path=few_shot_path,
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


def nll(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_questions,
        sampling_mode,
        k,
        aggregate_paths,
        multihop,
        nlp,
        few_shot,
        few_shot_path,
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

    paths_for_batch = _k_generation(
        model=model,
        tokenizer=tokenizer,
        batch_questions=batch_questions,
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
        do_sample=do_sample,
        multihop=multihop,
        nlp=nlp,
        few_shot=few_shot,
        few_shot_path=few_shot_path,
    )

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
