import nltk
import nltk.tokenize

class Tokenizer:
    """Tokenizer splits text into words.

    Args:
        - language: Text language.
        - download: If true, download required NLTK data packages during init.

    Inputs:
        - text: Input text.

    Outputs:
        Sequence of sentences, each sentence is a sequence of words.
        Each sentence ends with punctuation token.
    """
    def __init__(self, language="russian", download=True):
        if download:
            nltk.download("punkt", quiet=True)
        self._language = language

    def __call__(self, text):
        sentences = []
        for sentence in nltk.tokenize.sent_tokenize(text, language=self._language):
            sentences.append(list(map(str.lower, nltk.tokenize.word_tokenize(sentence, language=self._language))))
        return sentences
