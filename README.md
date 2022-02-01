<img src="img/nessie_with_text.svg" alt="Nessie Logo">

`nessie` is a package for annotation error detection. Recent work has shown that datasets used to evaluate
machine learning models still a considerable number of annotation errors. Annotation error detection can be
used to automatically detect these so that human annotators can concentrate on a subset to correct, instead
of needing to look at each and every instance.

## Installation

    pip install nessie

This installs the package with default dependencies and PyTorch with only CPU support.
If you want to use your own PyTorch version, you need to install it afterwards manually. 
If you need `faiss-gpu`, then you should also install that manually afterwards.

## Methods

We implement a wide range of annotation error detection methods. These are divided in two categories, *flaggers* and
*scorers*. *Flaggers* give a binary judgement whether an instance is considered wrong, *Scorers* give a certainty
estimate how likely it is that an instance is wrong.

## Models 

Model-based annotation detection methods need trained models to obtain predictions or probabilities.
We already implemented the most common models for you to be ready to use.
You can add your own models by implementing the respective abstract class for `TextClassifier` or `SequenceTagger`.
We provide the following models:

### Text classification

| Class name                | Description                                   |
|---------------------------|-----------------------------------------------|
| FastTextTextClassifier    | Fasttext                                      |
| FlairTextClassifier       | Flair                                         |
| LgbmTextClassifier        | LightGBM with handcrafted features            |
| LgbmTextClassifier        | LightGBM with S-BERT features                 |
| MaxEntTextClassifier      | Logistic Regression with handcrafted features |
| MaxEntTextClassifier      | Logistic with S-BERT features                 |
| TransformerTextClassifier | Transformers                                  |

You can easily add your own sklearn classifiers by subclassing `SklearnTextClassifier` like the following:

    class MaxEntTextClassifier(SklearnTextClassifier):
        def __init__(self, embedder: SentenceEmbedder, max_iter=10000):
            super().__init__(lambda: LogisticRegression(max_iter=max_iter, random_state=RANDOM_STATE), embedder)

### Sequence Classification

| Class name                | Description                   |
|---------------------------|-------------------------------|
| FlairSequenceTagger       | Flair                         |
| CrfSequenceTagger         | CRF with handcrafted features |
| MaxEntSequenceTagger      | Maxent sequence tagger        |
| TransformerSequenceTagger | Transformer                   |

## Development

We use [poetry](https://python-poetry.org/) for dependency management and packaging.
Follow their documentation to install it. Then you can run

    poetry install

to download the dependencies and install in its own environment.
In order to install your own PyTorch with CUDA, you can run

    poetry shell
    poe force-cuda113

or install it manually in the poetry environment. You can format the code via

    poe format

which should be run before every commit.