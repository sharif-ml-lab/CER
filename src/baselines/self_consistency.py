import torch

import gc
from transformers import PreTrainedModel, PreTrainedTokenizer
from src.utils import extract_last_numerical_value, postprocess_final_answer, extract_final_answer, construct_prompt


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
            tokenizer, batch_questions, multihop, device)

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

        for i in range(batch_size):
            generated_sequence = batch_output.sequences[i]
            input_length = tokenized_batch["input_ids"][i].shape[0]

            answer_ids = generated_sequence[input_length:]
            answer_text = tokenizer.decode(
                answer_ids, skip_special_tokens=True)

            batch_answers.append(answer_text)

            if not multihop:
                final_answer = postprocess_final_answer(
                    extract_last_numerical_value(answer_text))
            else:
                final_answer = extract_final_answer(answer_text)

            batch_final_answers.append(final_answer)

        for i in range(batch_size):
            answer_text = batch_answers[i]
            final_answer = batch_final_answers[i]
            paths[i].append(
                (answer_text, 1, final_answer))

    return paths


def batch_messages_creation(tokenizer, batch_questions, multihop, device):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    batch_messages = []
    for question in batch_questions:
        batch_messages.append([{"role": "user", "content": construct_prompt(
            question=question,
            few_shot=False,
            few_shot_path='',
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


def self_consistency_decode(
        model,
        tokenizer,
        batch_questions,
        k,
        multihop,
        num_beams=1,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
        length_penalty=1.0,
        no_repeat_ngram_size=0,
        max_new_tokens=1024,
        early_stopping=False,):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    do_sample = True

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
    )

    batch_results = []
    for path in paths_for_batch:
        voting = dict()
        for response in path:
            final_answer = response[2]
            if final_answer not in voting:
                voting[final_answer] = 1
            else:
                voting[final_answer] += 1

        best_answer = max(voting, key=voting.get)
        best_answer_confidence = voting[best_answer] / k
        batch_results.append([None, best_answer_confidence, best_answer])

    return batch_results
