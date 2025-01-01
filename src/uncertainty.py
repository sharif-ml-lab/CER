import torch

#  compute the confidence for a single numerical value occurrence in the generated answer.
def compute_confidence_for_value(output_scores, answer_ids, value_ids, device, confidence_method):
    value_start_idx = _find_subsequence_indices(answer_ids, value_ids, 1)
    if value_start_idx == -1:
        return None

    value_start_idx -= 1
    if value_start_idx < 0 or value_start_idx + len(value_ids) > len(output_scores):
        return None

    value_scores = output_scores[value_start_idx: value_start_idx + len(value_ids)]
    return calculate_confidence_for_final_answer(value_scores, torch.tensor(value_ids, device=device), confidence_method)

# calculate the confidence score (Δ) for the final answer tokens.
def calculate_confidence_for_final_answer(logits, answer_ids, confidence_method):
    confidence_sum = 1.0
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
                if confidence_method == "cot-decoding":
                    confidence_sum += (top_2_probs[-1][0] - top_2_probs[-1][1]).item()

                elif confidence_method == "weighted-difference":
                    confidence_sum += (top_2_probs[-1][0] - top_2_probs[-1][1]).item() * ans_token_prob.item()
                    confidence_sum += ans_token_prob.item()

                elif confidence_method == "selective":
                    confidence_sum *= ans_token_prob.item()

                elif confidence_method == "entropy":
                    # # Add a small epsilon to avoid log(0)
                    probs = torch.clamp(probs, min=1e-12)

                    # # Compute entropy: H(P) = -sum p_i * log(p_i)
                    # # Note: log is natural log by default in PyTorch
                    entropy = - (probs * torch.log(probs)).sum(dim=-1)
                    confidence_sum = 1 - entropy.item()

            else:
                confidence_sum += 1.0  # Only one token probability
        else:
            confidence_sum += 1.0  # Only one token probability
        valid_tokens += 1

    return confidence_sum / valid_tokens if valid_tokens > 0 else 0.0


#  find the start index of the Nth occurrence (occurrence_count) of subsequence in sequence. Returns -1 if not found.
def _find_subsequence_indices(sequence, subsequence, occurrence_count: int = 1):
    found_count = 0
    seq_len = len(sequence)
    sub_len = len(subsequence)

    for i in range(seq_len - sub_len + 1):
        if sequence[i:i + sub_len] == subsequence:
            if found_count == occurrence_count - 1:
                return i
            found_count += 1
    return -1


#  aggregate multiple paths based on their confidence scores.
def aggregate_paths_based_on_scores(paths):
    answer_scores = {}
    best_full_ans = None
    best_full_ans_delta = -1
    for answer, delta, final_answer in paths:
        answer_scores[final_answer] = answer_scores.get(final_answer, 0) + delta
        if best_full_ans_delta < delta:
            best_full_ans_delta = delta
            best_full_ans = answer
    best_answer = max(answer_scores, key=answer_scores.get)
    return best_full_ans, answer_scores[best_answer], best_answer