from setuptools import setup, find_namespace_packages


setup(
    version="0.0.1",
    name="pyfillet",
    long_description="Simple tokenizer and word2vec for Russian text.",
    url="https://github.com/ivan-chai/pyfillet",
    author="Ivan Karpukhin",
    author_email="karpuhini@yandex.ru",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "gensim>=4.2.0",
        "pymorphy2>=0.9.1",
        "nltk>=3.7"
    ]
)
