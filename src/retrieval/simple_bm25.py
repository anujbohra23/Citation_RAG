import math
from collections import Counter
from typing import List


class BM25Okapi:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / self.N if self.N else 0.0
        self.doc_freqs = []
        self.df = Counter()

        for doc in corpus:
            freqs = Counter(doc)
            self.doc_freqs.append(freqs)
            for term in freqs:
                self.df[term] += 1

        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.N
        if not query_tokens:
            return scores

        query_terms = Counter(query_tokens)
        for term, _ in query_terms.items():
            idf = self.idf.get(term, 0.0)
            for i, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                dl = self.doc_lens[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl > 0 else 1.0
                scores[i] += idf * (tf * (self.k1 + 1)) / denom
        return scores
