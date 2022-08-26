from unittest import TestCase, main

from pyfillet.embedder import *


class TestEmbedder(TestCase):
    def test_simple(self):
        embedder = Embedder()
        embedding = embedder("Мама")
        self.assertEqual(embedding.shape, (embedder.dim,))
        self.assertTrue(embedder("abrakadabra") is None)


if __name__ == "__main__":
    main()
