import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


class LanguageModelWrapper:
    """
    A wrapper class for a Hugging Face causal language model (LM).
    Provides methods to load the model and generate an answer.
    """

    def __init__(self, model_name: str, device: str = 'cpu'):
        """
        model_name: Model checkpoint from Hugging Face Hub, e.g. 'EleutherAI/gpt-neo-1.3B'.
        device: 'cpu' or 'cuda'
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate_answer(self, prompt: str, max_new_tokens: int = 100) -> dict:
        """
        Generates an answer using the model.

        Returns a dictionary containing:
          - 'answer': the generated string
          - 'log_prob': a mock or approximate log-probability (for demonstration)
        """

        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # using greedy decoding for reproducibility
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated_sequence = output.sequences[0]
        generated_text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)

        answer_part = generated_text[len(prompt):].strip()
        answer_ids = self.tokenizer.encode(answer_part)
        output_scores = output.scores

        return {
            'answer': answer_part,
            'answer_ids': answer_ids,
            'output_scores': output_scores
        }
