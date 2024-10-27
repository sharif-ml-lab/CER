import numpy as np
from replug.llm_client import query_lm


def reformulate_inputs(context, retrieved_docs):
    weights = []
    responses = []
    for doc in retrieved_docs:
        weighted_context = f"{doc} {context}"
        response = query_lm(weighted_context)
        responses.append(response)
        # Calculate similarity as a weighting factor
        weights.append(len(doc.split()))  # Dummy weighting based on length; replace with proper score

    # Weighted average ensemble
    weighted_responses = [response * weight for response, weight in zip(responses, weights)]
    final_output = sum(weighted_responses) / np.sum(weights)
    return final_output
