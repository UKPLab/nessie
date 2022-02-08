<img src="img/nessie_with_text.svg" alt="Nessie Logo">

**nessie** is a package for annotation error detection. It can be used to automatically detect errors in annotated
corpora so that human annotators can concentrate on a subset to correct, instead of needing to look at each
and every instance.

## Installation

    pip install nessie

This installs the package with default dependencies and PyTorch with only CPU support.
If you want to use your own PyTorch version, you need to install it afterwards manually. 
If you need `faiss-gpu`, then you should also install that manually afterwards.

## Methods

We implement a wide range of annotation error detection methods. These are divided in two categories, *flaggers* and
*scorers*. *Flaggers* give a binary judgement whether an instance is considered wrong, *Scorers* give a certainty
estimate how likely it is that an instance is wrong.

### Flagger
| **Abbreviation** | **Method**           | **Text** | **Token** | **Span** | **Proposed by**                                            |
|------------------|----------------------|----------|-----------|----------|------------------------------------------------------------|
| CL               | Confident Learning   | ✓        | ✓         | ✓        | Northcutt (2021)                                           | 
| CS               | Curriculum Spotter   | ✓        |           |          | Amiri (2018)                                               |
| DE               | Diverse Ensemble     | ✓        | ✓         | ✓        | Loftsson (2009)                                            |
| IRT              | Item Response Theory | ✓        | ✓         | ✓        | Rodriguez (2021)                                           | 
| LA               | Label Aggregation    | ✓        | ✓         | ✓        | Amiri (2018)                                               |
| LS               | Leitner Spotter      | ✓        |           |          | Amiri (2018)                                               |
| PE               | Projection Ensemble  | ✓        | ✓         | ✓        | Reiss (2020)                                               |
| RE               | Retag                | ✓        | ✓         | ✓        | van Halteren (2000)                                        |    
| VN               | Variation n-Grams    |          | ✓         | ✓        | Dickinson (2003)                                           | 

### Scorer

| **Abbreviation** | **Method**                 | **Text** | **Token** | **Span** | **Proposed by**                                            |
|------------------|----------------------------|----------|-----------|----------|------------------------------------------------------------|
| BC               | Borda Count                | ✓        | ✓         | ✓        | Larson (2020)                                              | 
| CU               | Classification Uncertainty | ✓        | ✓         | ✓        | Hendrycks (2017)                                           | 
| DM               | Data Map Confidence        | ✓        |           |          | Swayamdipta (2020)                                         |   
| DU               | Dropout Uncertainty        | ✓        | ✓         | ✓        | Amiri (2018)                                               |
| KNN              | k-Nearest Neighbor Entropy | ✓        | ✓         | ✓        | Grivas (2020)                                              |
| LE               | Label Entropy              |          | ✓         | ✓        | Hollenstein (2016)                                         |   
| MD               | Mean Distance              | ✓        | ✓         | ✓        | Larson (2019)                                              |
| PM               | Prediction Margin          | ✓        | ✓         | ✓        | Dligach (2011)                                             | 
| WD               | Weighted Discrepancy       |          | ✓         | ✓        | Hollenstein (2016)                                         |   rer

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

## Bibliography

**Amiri, Hadi, Timothy Miller, and Guergana Savova. 2018.**
"Spotting Spurious Data with Neural Networks."
*Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), 2006-16. New Orleans, Louisiana.*

**Dligach, Dmitriy, and Martha Palmer. 2011.**
"Reducing the Need for Double Annotation."
*Proceedings of the 5th Linguistic Annotation Workshop, 65-73. Portland, Oregon, USA.*

**Grivas, Andreas, Beatrice Alex, Claire Grover, Richard Tobin, and William Whiteley. 2020.**
"Not a Cute Stroke: Analysis of Rule- and Neural Network-based Information Extraction Systems for Brain Radiology Reports."
*Proceedings of the 11th International Workshop on Health Text Mining and Information Analysis, 24-37. Online.*

**Hendrycks, Dan, and Kevin Gimpel. 2017.**
"A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks."
*Proceedings of International Conference on Learning Representations, 1-12.*

**Hollenstein, Nora, Nathan Schneider, and Bonnie Webber. 2016.**
"Inconsistency Detection in Semantic Annotation."
*Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16), 3986-90. Portorož, Slovenia.*

**Larson, Stefan, Anish Mahendran, Andrew Lee, Jonathan K. Kummerfeld, Parker Hill, Michael A. Laurenzano, Johann Hauswald, Lingjia Tang, and Jason Mars. 2019.**
"Outlier Detection for Improved Data Quality and Diversity in Dialog Systems."
*Proceedings of the 2019 Conference of the North, 517-27. Minneapolis, Minnesota.*

**Loftsson, Hrafn. 2009.**
"Correcting a POS-Tagged Corpus Using Three Complementary Methods."
*Proceedings of the 12th Conference of the European Chapter of the ACL (EACL 2009), 523-31. Athens, Greece.*

**Northcutt, Curtis, Lu Jiang, and Isaac Chuang. 2021.**
"Confident Learning: Estimating Uncertainty in Dataset Labels."
*Journal of Artificial Intelligence Research 70 (April): 1373-1411.*

**Reiss, Frederick, Hong Xu, Bryan Cutler, Karthik Muthuraman, and Zachary Eichenberger. 2020.**
"Identifying Incorrect Labels in the CoNLL-2003 Corpus."
*Proceedings of the 24th Conference on Computational Natural Language Learning, 215-26. Online.*

**Rodriguez, Pedro, Joe Barrow, Alexander Miserlis Hoyle, John P. Lalor, Robin Jia, and Jordan Boyd-Graber. 2021.**
"Evaluation Examples Are Not Equally Informative: How Should That Change NLP Leaderboards?"
*Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 4486-4503. Online.*

**Swayamdipta, Swabha, Roy Schwartz, Nicholas Lourie, Yizhong Wang, Hannaneh Hajishirzi, Noah A. Smith, and Yejin Choi. 2020.**
"Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics."
*Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 9275-93. Online.*

**van Halteren, Hans. 2000.**
"The Detection of Inconsistency in Manually Tagged Text."
*Proceedings of the COLING-2000 Workshop on Linguistically Interpreted Corpora, 48-55. Luxembourg.*