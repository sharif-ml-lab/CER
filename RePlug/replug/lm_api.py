import openai


def query_lm(input_text):
    response = openai.Completion.create(
        engine="davinci",
        prompt=input_text,
        max_tokens=150
    )
    return response.choices[0].text


def get_lm_likelihood(context, continuation):
    # Send request to LM to get the likelihood for a given continuation
    response = openai.Completion.create(
        engine="davinci",
        prompt=context,
        max_tokens=len(continuation.split()),
        logprobs=1
    )
    return sum(response.choices[0].logprobs.token_logprobs)
