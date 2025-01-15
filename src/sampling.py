import torch

from src.uncertainty import _handle_all_decoding


def sampling_strategy(tokenizer, top_token_strs, top_token_ids, next_token_logits, generated_ids, output_scores, original_length, batch_sample_id, temperature=1.0, sampling_strategy="greedy_number_sampling"):
    if top_token_strs[batch_sample_id] == tokenizer.eos_token_id:
        return tokenizer.eos_token_id

    if sampling_strategy == "greedy_number_sampling":
        if top_token_strs[batch_sample_id].strip().isnumeric():
            chosen_token_id = top_token_ids[batch_sample_id]
        else:
            adjusted_logits = next_token_logits[batch_sample_id] / temperature
            adjusted_prob = torch.softmax(adjusted_logits, dim=-1)
            sampled_id = torch.multinomial(
                adjusted_prob, num_samples=1)
            chosen_token_id = sampled_id.item()

    elif sampling_strategy == "entropy_sampling":
        adjusted_logits = next_token_logits[batch_sample_id] / temperature
        adjusted_prob = torch.softmax(adjusted_logits, dim=-1)
        probs = torch.clamp(adjusted_prob, min=1e-12)
        entropy = - (probs * torch.log(probs)).sum(dim=-1)
        if entropy.item() > 0.5:
            sampled_id = torch.multinomial(adjusted_prob, num_samples=1)
            chosen_token_id = sampled_id.item()
        else:
            chosen_token_id = top_token_ids[batch_sample_id]

    elif sampling_strategy == "confidence_sampling":  # calculate the confidence till now
        answer_ids = generated_ids[batch_sample_id, original_length:]
        answer_text = tokenizer.decode(
            answer_ids, skip_special_tokens=True)

        result = _handle_all_decoding(tokenizer, "cuda", answer_text, output_scores, answer_ids, scoring_mode="log",
                                      confidence_method="default", multihop=False, doc=None, random_selection=False, random_selection_number_words=None)

        if result == None:
            confidence = 1.0
        else:
            confidence = result[1]

        if confidence < 0.5:
            chosen_token_id = top_token_ids[batch_sample_id]
        else:
            adjusted_logits = next_token_logits[batch_sample_id] / temperature
            adjusted_prob = torch.softmax(adjusted_logits, dim=-1)
            sampled_id = torch.multinomial(
                adjusted_prob, num_samples=1)
            chosen_token_id = sampled_id.item()

    return chosen_token_id
