from tests.fixtures import PATH_EXAMPLE_DATA_TEXT, PATH_EXAMPLE_DATA_TOKEN, PATH_EXAMPLE_DATA_SPAN

from nessie.dataloader import load_text_classification_tsv, load_sequence_labeling_dataset


def test_loading_text_classification_dataset():
    ds = load_text_classification_tsv(PATH_EXAMPLE_DATA_TEXT)

    assert len(ds.texts) == len(ds.gold_labels) == len(ds.noisy_labels)
    assert len(ds.texts) == 1536


def test_loading_token_labeling_dataset():
    ds = load_sequence_labeling_dataset(PATH_EXAMPLE_DATA_TOKEN)

    assert len(ds.sentences) == len(ds.gold_labels) == len(ds.noisy_labels)
    assert len(ds.sentences) == 3648


def test_loading_span_labeling_dataset():
    ds = load_sequence_labeling_dataset(PATH_EXAMPLE_DATA_SPAN)

    assert len(ds.sentences) == len(ds.gold_labels) == len(ds.noisy_labels)
    assert len(ds.sentences) == 701

