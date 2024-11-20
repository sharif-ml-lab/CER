from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, TopKLogitsWarper
import torch
import requests
import json


class HuggingFaceClient():
    def __init__(self, model_name: str):
        """
        Initialize the HuggingFace client with a specific model.

        Args:
            model_name (str): The name of the HuggingFace model to be used.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda', torch_dtype=torch.bfloat16)
        self.model.eval()
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

    async def next_prob(self, input_ids: torch.Tensor, prob_mode: bool):
        """
        Call the HuggingFace model with the provided input.

        Args:
            input_text (Tensor): The input text to be provided to the model.
            prob_mode (bool): A flag to specify whether to return the log probabilities.

        Returns:
            Either the string output or log probabilities, depending on the prob_mode flag.
        """
        input_ids = input_ids.to(self.device)
        if prob_mode:
            with torch.no_grad():
                outputs = self.model(input_ids, output_attentions=False, output_hidden_states=False)
                logits = outputs.logits

                # Calculate probabilities using softmax for the last token
                probabilities = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)

            return probabilities.detach().cpu().float().numpy()

        else:
            with torch.no_grad():
                # Generate output text
                generated_ids = self.model.generate(input_ids, max_length=input_ids.shape[1] + 50)
                return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def next_token_id(self, probs):
        # Get the most probable next token (greedy approach)
        next_token_id = torch.argmax(torch.from_numpy(probs), dim=-1)
        return next_token_id

    def decode_result(self, result_tokens):
        res = torch.tensor(result_tokens)
        return self.tokenizer.decode(res, skip_special_tokens=True)
