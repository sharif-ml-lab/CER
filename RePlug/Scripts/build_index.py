import torch
import faiss
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


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

    # Prepare FAISS index
    dimension = model.config.hidden_size
    index = faiss.IndexFlatL2(dimension)

    # Define batch size
    batch_size = 2048
    all_chunks = []

    # Iterate through the dataset and create text chunks
    for record in tqdm(dataset, desc="Tokenizing and chunking text"):
        text = record["text"]
        tokens = tokenizer.tokenize(text)
        chunks = [tokens[i:i + 128] for i in range(0, len(tokens), 128)]
        chunk_texts = [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]
        all_chunks.extend(chunk_texts)

    # Process chunks in batches to create embeddings and add to FAISS index
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding and indexing batches"):
        batch_chunks = all_chunks[i:i + batch_size]
        embeddings = embed_text_batch(batch_chunks, tokenizer, model)
        index.add(embeddings)

    # Save the FAISS index for future retrieval
    faiss.write_index(index, "/home/aquasar/Desktop/Ideas/data/wiki_embeddings.index")

    print("Indexing complete!")
