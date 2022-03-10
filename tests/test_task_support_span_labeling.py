import awkward as ak
from seqeval.metrics.sequence_labeling import get_entities
from sklearn.preprocessing import LabelEncoder

from nessie.dataloader import load_example_span_classification_data
from nessie.models.featurizer import FlairTokenEmbeddingsWrapper
from nessie.task_support.span_labeling import (
    SpanId,
    align_for_span_labeling,
    embed_spans,
    span_matching,
)


# Test alignment


def test_aggregate_result_to_spans():
    noisy_labels = [["B-PER", "I-PER", "O"], ["O", "B-ORG"]]
    predictions = [["B-PER", "I-PER", "O"], ["B-ORG", "I-ORG"]]
    probabilities = [
        # O|B-PER|I-PER|B-ORG|I-ORG
        [[0.1, 0.7, 0.1, 0.1, 0.0], [0.0, 0.2, 0.7, 0.0, 0.1], [0.9, 0.0, 0.1, 0.0, 0.0], [0.9, 0.0, 0.0, 0.0, 0.1]],
        [[0.1, 0.1, 0.1, 0.7, 0.0], [0.1, 0.1, 0.1, 0.0, 0.7]],
    ]

    # TODO: Test repeated probabilities
    repeated_probabilities = None
    le = LabelEncoder()
    le.classes_ = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"]

    result = align_for_span_labeling(noisy_labels, predictions, probabilities, repeated_probabilities, le)

    assert list(result.labels) == ["PER", "ORG"]
    assert result.span_ids == [SpanId(0, 0, 2), SpanId(1, 1, 2)]
    assert len(result.le.classes_) == 3


# Test span matching


def test_span_matching():
    a = [(0, 2), (4, 5), (8, 10)]
    b = [(0, 2), (3, 5), (5, 6), (7, 9)]

    # [0,0]: identical span
    # [1,1]: span in a is a subset of b
    # then we omit 2 from b, which has no corresponding span in a
    # [2,3]: spans in a and b intersect, but none is a subset of the other
    expected = {0: 0, 1: 1, 2: 3}

    actual = span_matching(a, b)
    assert actual == expected


def test_span_matching_keep_a():
    a = [(5, 7), (8, 11), (13, 15), (16, 18), (18, 21), (22, 28)]
    b = [(0, 3), (5, 7), (12, 14), (16, 21), (22, 25), (26, 28)]

    # [0,1]: identical span
    # [1,None]: only in A
    # [2,2]: spans in a and b intersect, but none is a subset of the other
    # [4,3]: one span from B encompasses two spans from A, 4 is the one with greater overlap from A
    # [5,4]: one span from A encompasses two spans from B, 4 is the one with greater overlap from B
    expected = {0: 1, 1: None, 2: 2, 3: None, 4: 3, 5: 4}

    actual = span_matching(a, b, keep_A=True)
    assert actual == expected


# Embedding


def test_embed_spans(token_embedder_fixture: FlairTokenEmbeddingsWrapper):
    n = 100
    ds = load_example_span_classification_data().subset(n)

    embedded_spans = embed_spans(ds.sentences, ds.noisy_labels, token_embedder_fixture)

    entities = get_entities(ds.noisy_labels.tolist())

    assert len(embedded_spans) == n
    assert len(ak.flatten(embedded_spans)) == len(entities)
