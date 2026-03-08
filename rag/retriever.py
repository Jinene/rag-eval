from .embeddings import embed
from .vector_store import VectorStore

class Retriever:

    def __init__(self, documents):
        self.documents = documents
        doc_embeddings = embed(documents)
        self.store = VectorStore(doc_embeddings)

    def retrieve(self, query):

        q = embed([query])
        indices = self.store.search(q)

        return [self.documents[i] for i in indices[0]]
