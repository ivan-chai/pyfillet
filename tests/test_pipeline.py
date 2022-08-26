from unittest import TestCase, main

from pyfillet.pipeline import *


class TestTextEmbedder(TestCase):
    def test_simple(self):
        text = "Мама мыла раму. Папа, к.м.н., сажал забор?"
        embedder = TextEmbedder()
        embeddings = embedder(text)
        self.assertEqual(len(embeddings), 8)
        self.assertEqual(len(embeddings[0]), 2)
        self.assertEqual(embeddings[0][0], "Мама")
        self.assertEqual(embeddings[0][1].shape, (embedder.dim,))


if __name__ == "__main__":
    main()
