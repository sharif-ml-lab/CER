import faiss
import numpy as np
import pickle


class FaissRetriever:
    def __init__(self, index_path, chunks_path, metric='L2'):
        # Load the FAISS index from the given path
        self.index = faiss.read_index(index_path)

        # Set the metric type for the index
        if metric == 'cosine':
            self.metric = 'cosine'
            # Normalize all vectors in the index
            self._normalize_index_vectors()
        else:
            self.metric = 'L2'

        # Load the index and chunk texts for retrieval
        with open(chunks_path, "rb") as f:
            self.chunk_texts = pickle.load(f)

    def _normalize_index_vectors(self):
        # Get all vectors from the index and normalize them
        vectors = self.index.reconstruct_n(0, self.index.ntotal)
        faiss.normalize_L2(vectors)
        # Create a new index and add the normalized vectors
        dimension = vectors.shape[1]
        normalized_index = faiss.IndexFlatIP(dimension)  # Use IndexFlatIP for cosine similarity
        normalized_index.add(vectors)
        self.index = normalized_index

    def retrieve(self, vector, top_n=10):
        # Ensure the vector is in the correct shape for searching
        if len(vector.shape) == 1:
            vector = np.expand_dims(vector, axis=0)

        # If using cosine similarity, normalize the query vector
        if self.metric == 'cosine':
            faiss.normalize_L2(vector)

        # Search for the top_n closest indices
        distances, indices = self.index.search(vector, top_n)
        results = [self.chunk_texts[idx] for idx in indices[0]]
        return results, distances


# Example usage
if __name__ == "__main__":
    # Create an instance of FaissRetriever with the previously saved index
    index_path = "wiki_embeddings.index"
    retriever = FaissRetriever(index_path, metric='cosine')

    # Example query vector (must be the same dimension as used in the index)
    query_vector = np.random.random((768,)).astype('float32')  # Replace with a real embedding

    # Retrieve the top 5 closest vectors
    indices, distances = retriever.retrieve(query_vector, top_n=5)

    print("Top 5 closest indices:", indices)
    print("Corresponding distances:", distances)
