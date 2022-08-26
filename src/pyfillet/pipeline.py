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
        """Embedding dimension."""
        return self._embedder.dim

    @property
    def embeddings(self):
        """Dictionary of embeddings for the words."""
        return self._embedder.embeddings

    def __call__(self, text):
        results = []
        for sentence in self._tokenizer(text):
            results.append(list(filter(lambda word_embedding: word_embedding[1] is not None, zip(sentence, map(self._embedder, sentence)))))
            if (sentence[-1] in [".", "?", "!"]) and (self._eos_embedding is not None):
                results.append([(sentence[-1], self._eos_embedding)])
        return sum(results, [])
