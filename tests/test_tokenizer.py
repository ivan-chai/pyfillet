from unittest import TestCase, main

from pyfillet.tokenizer import *


class TestTokenizer(TestCase):
    def test_simple(self):
        sentences = Tokenizer()("Мама мыла раму. Папа к.м.н. сажал забор?")
        gt_output = [["Мама", "мыла", "раму", "."], ["Папа", "к.м.н.", "сажал", "забор", "?"]]
        self.assertEqual(sentences, gt_output)


if __name__ == "__main__":
    main()
