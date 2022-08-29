import os
import logging
import urllib.parse
import urllib.request
import zipfile

import gensim
import pymorphy2

from .utils import md5, DownloadBar


class WordEmbedder:
    """Embedder computes embeddings for the words.

    Args:
        - model: Word2vec model name.
        - root: Path to save model (default is local cache directory).
        - download: Download the model if it is missing or broken.

    Inputs:
        - word: Input word.

    Outputs:
        Word embedding or `None` if word was not found in the dictionary.
    """

    MODELS = {
        "rusvectores-180": {
            "url": ("http://vectors.nlpl.eu/repository/20/180.zip", "aa919ce69a5a12f8d02fe4f2751d67aa"),
            "model": ("model.bin", "8825f9a42305cdcc1af11d3acde53280")
        },
        "rusvectores-220": {
            "url": ("http://vectors.nlpl.eu/repository/20/220.zip", "dd315d3e317cecf4826aeb42546d3326"),
            "model": ("model.bin", "2f0eaace5eb2e9ab8f89e994d1f724c3")
        }
    }

    POS_MAP = {
        "ADJF": "ADJ",
        "ADJS": "ADJ",
        "ADVB": "ADV",
        "COMP": "ADV",
        "GRND": "VERB",
        "INFN": "VERB",
        "PRED": "ADV",
        "PRTF": "ADJ",
        "PRTS": "VERB"
    }

    def __init__(self, model="rusvectores-220", root=None, download=True):
        if root is None:
            cache = os.path.expanduser(os.path.join("~", ".cache"))
            if download and not os.path.isdir(cache):
                os.mkdir(cache)
            root = os.path.join(cache, "pyfillet")

        meta = self.MODELS[model]
        url, url_md5 = meta["url"]
        model_filename, model_md5 = meta["model"]
        filename = os.path.basename(urllib.parse.urlparse(url).path)
        path = os.path.join(root, filename)
        model_root = os.path.splitext(path)[0]
        model_path = os.path.join(model_root, model_filename)

        if not os.path.isfile(model_path) or md5(model_path) != model_md5:
            if not download:
                raise FileNotFoundError("Can't find word2vec model or MD5 mismatch ({})".format(model_path))
            if not os.path.isdir(root):
                os.mkdir(root)
            if not os.path.isfile(path) or md5(path) != url_md5:
                logging.info("Download model from {}".format(url))
                pbar = DownloadBar()
                try:
                    urllib.request.urlretrieve(url, path, reporthook=pbar)
                except Exception:
                    if os.path.isfile(path):
                        os.remove(path)
                finally:
                    pbar.close()
            with zipfile.ZipFile(path, "r") as zfp:
                zfp.extractall(model_root)
        self._model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
        self._model.fill_norms()
        self._morpher = pymorphy2.MorphAnalyzer()

    @property
    def dim(self):
        """Embedding dimension."""
        return self._model.vector_size

    @property
    def embeddings(self):
        """Dictionary of embeddings for the words."""
        return {key: self._model.get_vector(index) for key, index in self._model.key_to_index.items()}

    def __call__(self, word):
        word = word.lower()
        parse_result = self._morpher.parse(word)
        if len(parse_result) == 0:
            return
        parse_result = parse_result[0]
        pos = parse_result.tag.POS
        if pos is None:
            return
        pos = self.POS_MAP.get(pos, pos)
        lemma = parse_result.normal_form
        for token in [word + "_" + pos, lemma + "_" + pos, word + "_" + "PROPN", lemma + "_" + "PROPN"]:
            index = self._model.key_to_index.get(token)
            if index is not None:
                break
        else:
            return
        return self._model.get_vector(index)
