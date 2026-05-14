import numpy as np

class SemanticCache:
    def __init__(self, embedding_model, threshold=0.85):
        self.embedding_model = embedding_model
        self.cached_entries  = []
        self.threshold       = threshold

    def set(self, query: str, answer: str, route: str):
        for entry in self.cached_entries:
            if entry['query'] == query:
                return
        embedded_query = self.embedding_model.embed_query(query)
        set_dict = {'query': query, 'answer': answer, 
                    'route': route, 'embedded_query': embedded_query}
        self.cached_entries.append(set_dict)

    def get(self, query: str) -> str | None:
        for entry in self.cached_entries:
            if entry['query'] == query:
                return entry['answer']
        query_embedding = np.array(self.embedding_model.embed_query(query))
        best_score = 0
        best_entry = None
        for entry in self.cached_entries:
            cached_embedding = np.array(entry['embedded_query'])
            similarity = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )
            if similarity > best_score:
                best_score = similarity
                best_entry = entry
        if best_score > self.threshold:
            return best_entry['answer']
        return None