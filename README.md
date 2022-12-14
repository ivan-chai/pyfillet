# pyfillet
<p align="center">
<img src="https://github.com/ivan-chai/pyfillet/blob/main/docs/levitan.jpg?raw=true" alt="Levitan" width="75%"/>
</p>

Simple tokenizer and word2vec for Russian text. The package can be used to preprocess and embed text for LSTM or other language model.

The name of the pacckage refers to picture framing. According to Wikipedia:

> A *fillet* (also referred to as a slip) is a small piece of moulding which fits inside a larger frame or, typically, underneath or in between matting, used for decorative purposes.

Example code:
```python
from pyfillet import TextEmbedder

text = "Мама мыла раму. Папа сажал забор?"

embedder = TextEmbedder()
embeddings = embedder(text)
print(len(embeddings))        # "8", the number of encoded words and sentence endings.
print(embeddings[0][0])       # "мама", the first word.
print(len(embeddings[0][1]))  # "300", an embedding dim.
```