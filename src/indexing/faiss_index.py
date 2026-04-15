class FaissIndex:
    def __init__(self, *args, **kwargs):
        self.initialized = False

    def build(self, embeddings):
        raise NotImplementedError('Implement in Weeks 7-8.')

    def search(self, query_embedding, top_k: int = 10):
        raise NotImplementedError('Implement in Weeks 7-8.')
