import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Tuple

from src.uncertainty import aggregate_paths_based_on_scores


def calculate_confidence_for_final_answer(
        logits: List[torch.Tensor],
        answer_ids: torch.Tensor
) -> float:
    """
    Calculate the confidence score for the final answer tokens.
    """
    confidence_sum = 0.0
    valid_tokens = 0
    for t, token_id in enumerate(answer_ids):
        if t >= len(logits):
            break
        token_logits = logits[t]
        probs = torch.softmax(token_logits, dim=-1)
        ans_token_prob = probs[-1][token_id]
        if probs.size(-1) > 1:
            top_2_probs, _ = torch.topk(probs, min(2, probs.size(-1)))
            if top_2_probs.size(-1) > 1:
                # Here we add the probability of the answer token to the confidence
                confidence_sum += ans_token_prob.item()
            else:
                # Only one token probability
                confidence_sum += 1.0
        else:
            # Only one token probability
            confidence_sum += 1.0
        valid_tokens += 1

    return confidence_sum / valid_tokens if valid_tokens > 0 else 0.0


def _sample_cot_paths(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        k: int,
        max_new_tokens: int,
        temperature: float
) -> List[Tuple[str, float, str]]:
    """
    Generate paths using CoT sampling mode.
    We pick the top-k next tokens, then generate one path per token.
    """
    paths = []
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        first_token_logits = outputs.logits[0, -1, :]
        _, top_k_indices = torch.topk(first_token_logits, k)

    for idx in top_k_indices:
        generated_ids = torch.cat(
            [input_ids, idx.unsqueeze(0).unsqueeze(0)], dim=-1)
        scores_list = []

        gen_mask = torch.cat(
            [attention_mask, torch.ones(
                (1, 1), dtype=torch.long, device=device)],
            dim=-1
        )

        # Start generation loop
        for _ in range(max_new_tokens):
            outputs = model(generated_ids, attention_mask=gen_mask)
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            probabilities = torch.softmax(next_token_logits, dim=-1)
            scores_list.append(next_token_logits.detach().cpu())

            # Highest probability token
            top_token_id = torch.argmax(probabilities, dim=-1).item()
            top_token_str = tokenizer.decode([top_token_id])

            # Stop if EOS token is generated
            if top_token_id == tokenizer.eos_token_id:
                break

            # Check if the top token is numeric; if not, apply temperature sampling
            if top_token_str.isnumeric():
                next_token_id = top_token_id
            else:
                adjusted_logits = next_token_logits / temperature
                adjusted_probabilities = torch.softmax(adjusted_logits, dim=-1)
                next_token_id = torch.multinomial(
                    adjusted_probabilities, num_samples=1).item()

            generated_ids = torch.cat(
                [generated_ids, torch.tensor(
                    [[next_token_id]], dtype=torch.long, device=device)],
                dim=1
            )
            gen_mask = torch.cat(
                [gen_mask, torch.ones(
                    (1, 1), dtype=torch.long, device=device)],
                dim=-1
            )

        answer_text = tokenizer.decode(
            generated_ids[0], skip_special_tokens=True)
        output_scores = scores_list

        result = _handle_new_decoding_cot_mode(
            tokenizer, device, answer_text, output_scores, generated_ids[0]
        )

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
        temperature: float,
        scoring_mode: str
) -> List[Tuple[str, float, str]]:
    """
    Generate paths using temperature-based sampling mode.
    We call model.generate k times, each time generating a single path.
    """
    paths = []

    for _ in range(k):
        generated_ids = input_ids.clone()
        scores_list = []

        gen_mask = attention_mask

        for _ in range(max_new_tokens):
            outputs = model(generated_ids, attention_mask=gen_mask)
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            probabilities = torch.softmax(next_token_logits, dim=-1)
            scores_list.append(next_token_logits.detach().cpu())

            top_token_id = torch.argmax(probabilities, dim=-1).item()
            top_token_str = tokenizer.decode([top_token_id])

            if top_token_id == tokenizer.eos_token_id:
                break

            if top_token_str.isnumeric():
                next_token_id = top_token_id
            else:
                adjusted_logits = next_token_logits / temperature
                adjusted_probabilities = torch.softmax(adjusted_logits, dim=-1)
                next_token_id = torch.multinomial(
                    adjusted_probabilities, num_samples=1).item()

            generated_ids = torch.cat(
                [generated_ids, torch.tensor(
                    [[next_token_id]], dtype=torch.long, device=device)],
                dim=1
            )
            gen_mask = torch.cat(
                [gen_mask, torch.ones(
                    (1, 1), dtype=torch.long, device=device)],
                dim=-1
            )

        answer_text = tokenizer.decode(
            generated_ids[0], skip_special_tokens=True)
        output_scores = scores_list

        result = _handle_new_decoding_temp_mode(
            tokenizer, device, answer_text, output_scores, generated_ids[0], scoring_mode
        )

        if result is not None:
            paths.append(result)

    return paths


def greedy_number_cot_decode(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        messages,
        sampling_mode,
        scoring_mode,
        k,
        max_new_tokens,
        temperature,
        aggregate_paths,):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        input_text = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in messages])
        input_text += "\nassistant:"

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # # to set parameters for temperature-based sampling (different each time)
    # if sampling_mode == "temperature":
    #     do_sample = True

    # # to set parameters for greedy-based sampling (unique each time)
    # elif sampling_mode == "greedy":
    #     do_sample = False

    if baseline_cot == "k-branch":  # make k branches and then continue each one with sampling mode
        paths = _k_branch_generation(
            model, tokenizer, device, input_ids, attention_mask, k,
            max_new_tokens, num_beams, temperature, top_p,
            repetition_penalty, length_penalty, no_repeat_ngram_size,
            early_stopping, decoding_mode, scoring_mode, do_sample, confidence_method)

    elif baseline_cot == "k-seperate":  # make k distinict paths with sampling mode
        paths = _k_seperate_generation(
            model, tokenizer, device, input_ids, attention_mask, k,
            max_new_tokens, num_beams, temperature, top_p,
            repetition_penalty, length_penalty, no_repeat_ngram_size,
            early_stopping, decoding_mode, scoring_mode, do_sample, confidence_method)

    # if sampling_mode == "cot":
    #     paths = _sample_cot_paths(
    #         model=model,
    #         tokenizer=tokenizer,
    #         device=device,
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         k=k,
    #         max_new_tokens=max_new_tokens,
    #         temperature=temperature
    #     )
    # elif sampling_mode == "temp":
    #     paths = _sample_temp_paths(
    #         model=model,
    #         tokenizer=tokenizer,
    #         device=device,
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         k=k,
    #         max_new_tokens=max_new_tokens,
    #         temperature=temperature,
    #         scoring_mode=scoring_mode
    #     )
    else:
        raise ValueError("Unsupported sampling_mode")

    if not paths:
        return "", 0.0, ""

    if aggregate_paths:
        return aggregate_paths_based_on_scores(paths)
    else:
        return max(paths, key=lambda x: x[1])
