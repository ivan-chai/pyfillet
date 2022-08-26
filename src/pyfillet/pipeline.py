import numpy as np
from .tokenizer import Tokenizer
from .embedder import WordEmbedder


class TextEmbedder:
    """Encode text into a sequence of word embeddings.

    Args:
        language: Text language.

    Inputs:
        - text: Input text.

    Outputs:
        - List of (word, embedding) pairs.
    """
    def __init__(self, language="russian"):
        self._tokenizer = Tokenizer()
        self._embedder = WordEmbedder()
        has_punct = all([self._embedder(mark) is not None for mark in ".?!"])
        self._eos_embedding = np.zeros(self._embedder.dim) if not has_punct else None

    @property
    def dim(self):
        return self._embedder.dim

    def __call__(self, text):
        results = []
        for sentence in self._tokenizer(text):
            results.append(list(filter(lambda word_embedding: word_embedding[1] is not None, zip(sentence, map(self._embedder, sentence)))))
            if self._eos_embedding is not None:
                mark = sentence[-1] if sentence[-1] in [".", "?", "!"] else "."
                results.append([(mark, self._eos_embedding)])
        return sum(results, [])
