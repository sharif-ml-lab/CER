import torch

from src.utils import extract_last_numerical_value, postprocess_final_answer, extract_final_answer


class SelfConsistency:
    def __init__(self, model, tokenizer, num_samples):
        self.model = model
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.num_samples = num_samples

    def generate_responses(self, message: str):
        input_ids = self.tokenizer.encode(
            message, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)

        responses = []
        probs = []
        for _ in range(self.num_samples):
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                repetition_penalty=1.0,
                length_penalty=1.0,
                no_repeat_ngram_size=0,
                early_stopping=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

            generated_sequence = output.sequences[0]
            answer_ids = generated_sequence[len(input_ids[0]):]
            answer_text = self.tokenizer.decode(
                answer_ids, skip_special_tokens=True)
            probs.append(output.scores)

            responses.append(answer_text)
        return responses, probs

    def evaluate(self, prompt, multihop):
        responses, probs = self.generate_responses(prompt)

        voting = dict()
        for response in responses:

            if not multihop:
                final_answer = postprocess_final_answer(
                    extract_last_numerical_value(response))
            else:
                final_answer = extract_final_answer(response)

            if final_answer not in voting:
                voting[final_answer] = 1
            else:
                voting[final_answer] += 1

        best_answer = max(voting, key=voting.get)
        best_answer_confidenc = voting[best_answer] / self.num_samples
        return [None, best_answer_confidenc, best_answer]


def self_consistency_decode(
        model,
        tokenizer,
        messages,
        k, multihop):
    batch_results = []

    for message in messages:

        if tokenizer.chat_template:
            input_text = tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True)
        else:
            input_text = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in message])
            input_text += "\nassistant:"

        self_consistency = SelfConsistency(model, tokenizer, num_samples=k)
        result = self_consistency.evaluate(input_text, multihop)

        batch_results.append(result)

    return batch_results
