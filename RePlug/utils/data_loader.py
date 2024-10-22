import json

def load_training_data(filepath='training_data.json'):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def load_corpus(filepath='corpus.json'):
    with open(filepath, 'r') as f:
        corpus = json.load(f)
    return corpus