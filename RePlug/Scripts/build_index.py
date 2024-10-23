import torch
import faiss
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import deepspeed
import pickle


def embed_text_batch(chunks, tokenizer, model, device='cuda'):
    # Tokenize chunks as a batch
    inputs = tokenizer(chunks, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        # Get embeddings for the batch
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # mean pooling of the last hidden of the tokens
    return embeddings.detach().cpu().numpy()


if __name__ == '__main__':
    # Load the wiki dataset from HuggingFace
    dataset = load_from_disk('/home/aquasar/Desktop/Ideas/data/datasets/wiki.hf')
    dataset = dataset['train']

    # Load the tokenizer and model from Hugging Face
    model_name = "facebook/contriever"  # You can change this to any model you prefer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to('cuda')
    model.eval()

    # Initialize DeepSpeed-Inference
    ds_model = deepspeed.init_inference(model, mp_size=1)

    # Prepare FAISS index
    dimension = model.config.hidden_size
    index = faiss.IndexFlatL2(dimension)

    # Define batch size
    batch_size = 2048
    text_batch_size = 2048
    all_texts = []

    # Collect all texts from the dataset for batch tokenization
    for record in tqdm(dataset, desc="Collecting texts"):
        all_texts.append(record["text"])

    # Tokenize texts in batches
    all_chunks = []
    chunk_texts = []
    for i in tqdm(range(0, len(all_texts), text_batch_size), desc="Tokenizing texts in batches"):
        batch_texts = all_texts[i:i + text_batch_size]
        batch_tokens = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128,
                                 add_special_tokens=False)
        for j in range(len(batch_texts)):
            tokens = batch_tokens.input_ids[j]
            chunks = [tokens[k:k + 128] for k in range(0, len(tokens), 128)]
            chunk_texts_batch = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
            all_chunks.extend(chunk_texts_batch)
            chunk_texts.extend(chunk_texts_batch)

    # Process chunks in batches to create embeddings and add to FAISS index
    chunk_metadata = []
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding and indexing batches"):
        batch_chunks = all_chunks[i:i + batch_size]
        embeddings = embed_text_batch(batch_chunks, tokenizer, ds_model)
        index.add(embeddings)
        chunk_metadata.extend(batch_chunks)

    # Save the FAISS index for future retrieval
    faiss.write_index(index, "/home/aquasar/Desktop/Ideas/data/wiki_embeddings.index")

    # Save chunk metadata (for retrieval) using pickle
    with open("/home/aquasar/Desktop/Ideas/data/wiki_chunk_texts.pkl", "wb") as f:
        pickle.dump(chunk_texts, f)

    print("Indexing complete!")
