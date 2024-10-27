from abc import ABC, abstractmethod
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, TopKLogitsWarper
import torch
import requests
import json


# Abstract Base Class for LLM Client
class LLMClient(ABC):
    @abstractmethod
    def call(self, input_text: str, prob_mode: bool):
        """
        Call the LLM with the provided input.

        Args:
            input_text (str): The input text to be provided to the LLM.
            prob_mode (bool): A flag to specify whether to return the log probabilities.

        Returns:
            Depending on the prob_mode flag, either returns the string output or log probabilities.
        """
        pass


# OpenAI Client that Implements the LLMClient Abstract Class
class OpenAIClient(LLMClient):
    def __init__(self, model_name: str, api_key: str):
        """
        Initialize the OpenAI client with a specific model.

        Args:
            model_name (str): The name of the OpenAI model to be used.
            api_key (str): The API key to authenticate with OpenAI.
        """
        self.model_name = model_name
        openai.api_key = api_key

    async def call(self, input_text: str, prob_mode: bool):
        """
        Call the OpenAI model with the provided input.

        Args:
            input_text (str): The input text to be provided to the model.
            prob_mode (bool): A flag to specify whether to return the log probabilities.

        Returns:
            Either the string output or log probabilities, depending on the prob_mode flag.
        """
        response = await openai.Completion.acreate(
            model=self.model_name,
            prompt=input_text,
            max_tokens=150,
            logprobs=5 if prob_mode else None
        )
        if prob_mode:
            # Extract and return the log probabilities from the response
            return response.choices[0].logprobs
        else:
            # Extract and return the text output from the response
            return response.choices[0].text.strip()


# HuggingFace Client that Implements the LLMClient Abstract Class
class HuggingFaceClient(LLMClient):
    def __init__(self, model_name: str):
        """
        Initialize the HuggingFace client with a specific model.

        Args:
            model_name (str): The name of the HuggingFace model to be used.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    async def call(self, input_text: str, prob_mode: bool):
        """
        Call the HuggingFace model with the provided input.

        Args:
            input_text (str): The input text to be provided to the model.
            prob_mode (bool): A flag to specify whether to return the log probabilities.

        Returns:
            Either the string output or log probabilities, depending on the prob_mode flag.
        """
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=False, output_hidden_states=False)
            logits = outputs.logits

        if prob_mode:
            # Calculate probabilities using softmax
            probabilities = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
            log_probs = torch.log(probabilities)
            topk_probs, topk_indices = torch.topk(log_probs, k=5, dim=-1)
            return {
                "tokens": [self.tokenizer.decode([idx]) for idx in topk_indices[0]],
                "log_probs": topk_probs[0].tolist()
            }
        else:
            # Generate output text
            generated_ids = self.model.generate(input_ids, max_length=input_ids.shape[1] + 50)
            return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)



# Example Usage
if __name__ == "__main__":
    # OpenAI Client Example
    model_name = "text-davinci-003"
    api_key = "YOUR_OPENAI_API_KEY"
    client = OpenAIClient(model_name=model_name, api_key=api_key)
    input_text = "What is the capital of France?"
    output_text = client.call(input_text, prob_mode=False)
    print("OpenAI Output:", output_text)

    # HuggingFace Client Example
    hf_model_name = "gpt2"
    hf_client = HuggingFaceClient(model_name=hf_model_name)
    hf_output_text = hf_client.call(input_text, prob_mode=False)
    print("HuggingFace Output:", hf_output_text)
    hf_log_probs = hf_client.call(input_text, prob_mode=True)
    print("HuggingFace Log Probabilities:", hf_log_probs)
