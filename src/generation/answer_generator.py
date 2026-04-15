class AnswerGenerator:
    def __init__(self, *args, **kwargs):
        self.initialized = False

    def generate(self, query: str, contexts: list[dict]) -> str:
        raise NotImplementedError('Implement in Weeks 11-12.')
