import faiss
import numpy as np

class VectorStore:

    def __init__(self, embeddings):
        self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.add(np.array(embeddings))

    def search(self, query_vector, k=2):
        distances, indices = self.index.search(query_vector, k)
        return indices
