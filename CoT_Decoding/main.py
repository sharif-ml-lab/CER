# Usage example
from fasttext.util import download_model
from rpyc.utils.classic import download
from transformers import AutoModelForCausalLM, AutoTokenizer
from Decoding import get_device, cot_decode

model_name = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "user",
     "content": "In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What percentage of the entire students enrolled in hip-hop dance?"}
]

# Generate the response using CoT decoding
print(f"Using device: {get_device()}")
result, confidence = cot_decode(model, tokenizer, messages, aggregate_paths=True, max_new_tokens=512)
print(f"CoT Decoding:\n {result}")
