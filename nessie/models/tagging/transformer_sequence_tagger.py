import tempfile
import warnings
from typing import Dict, List, Optional

import awkward as ak
import more_itertools
import numpy as np
import seqeval.metrics
import torch
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder, normalize
from torch import nn
from torch.types import Device
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    IntervalStrategy,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from nessie.config import BERT_BASE
from nessie.models.model import Callbackable, SequenceTagger
from nessie.types import RaggedStringArray
from nessie.util import RANDOM_STATE, my_backoff


class TransformerSequenceTagger(SequenceTagger, Callbackable):
    # https://github.com/huggingface/transformers/blob/master/examples/pytorch/token-classification/run_ner.py
    # https://huggingface.co/transformers/custom_datasets.html?highlight=offset_mapping#token-classification-with-w-nut-emerging-entities

    @my_backoff()
    def __init__(
        self,
        verbose: bool = True,
        max_epochs: int = 24,
        batch_size: int = 32,
        model_name: str = BERT_BASE,
    ):
        self._verbose = verbose
        self._max_epochs = max_epochs
        self._batch_size = batch_size
        self._model_name = model_name

        self._callbacks: Dict[str, TrainerCallback] = {}

        self._model: Optional[PreTrainedModel] = None
        self._label_encoder: Optional[LabelEncoder] = None

        self._label_to_id: Optional[Dict[str, int]] = None
        self._id_to_label: Optional[Dict, int, str] = None

        self._is_roberta = "roberta" in model_name

        self._use_mc_dropout = False

        self._tokenizer = self._build_tokenizer()

    def add_callback(self, name: str, callback: TrainerCallback):
        self._callbacks[name] = callback

    @my_backoff()
    def fit(self, X: RaggedStringArray, y: RaggedStringArray):
        unique_labels = set(tag for tags in y for tag in tags)

        label_to_id = {}
        id_to_label = {}

        for i, tag in enumerate(sorted(unique_labels)):
            label_to_id[tag] = i
            id_to_label[i] = tag

        self._label_to_id = label_to_id
        self._id_to_label = id_to_label

        tokenized_stuff = self._tokenize_and_align_labels(X, y)

        model = self._build_model()
        self._model = model

        data_collator = DataCollatorForTokenClassification(self._tokenizer)

        train_dataset = SequenceTaggingTrainDataset(tokenized_stuff["input_ids"], tokenized_stuff["labels"])

        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        callbacks.extend(self._callbacks.values())

        with tempfile.TemporaryDirectory() as tmpdirname:
            training_args = TrainingArguments(
                output_dir=tmpdirname,
                num_train_epochs=self._max_epochs,
                per_device_train_batch_size=self._batch_size,
                per_device_eval_batch_size=64,
                save_strategy=IntervalStrategy.EPOCH,
                seed=RANDOM_STATE,
                save_total_limit=2,
                disable_tqdm=not self._verbose,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                evaluation_strategy=IntervalStrategy.EPOCH,
                logging_steps=200,
                logging_strategy=IntervalStrategy.EPOCH if self._verbose else IntervalStrategy.NO,
                log_level="info" if self._verbose else "error",
                report_to="none",
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=train_dataset,
                compute_metrics=self._compute_metrics,
                data_collator=data_collator,
                callbacks=callbacks,
            )

            train_result = trainer.train()

    def predict(self, X: RaggedStringArray) -> ak.Array:
        assert self._model, "Model not set for predicting, train first"

        proba = self.predict_proba(X)

        return ak.Array(
            [[self._id_to_label[np.argmax(token_probs)] for token_probs in sentence_probs] for sentence_probs in proba]
        )

    def score(self, X: RaggedStringArray) -> ak.Array:
        assert self._model, "Model not set for predicting, train first"

        proba = self.predict_proba(X)

        return ak.Array([[np.max(token_probs) for token_probs in sentence_probs] for sentence_probs in proba])

    def predict_proba(self, X: RaggedStringArray) -> ak.Array:
        assert self._model, "Model not set for predicting, train first"

        self._model.eval()
        self._activate_mc_dropout_if_needed()

        tokenizer = self._tokenizer

        X = [list(s) for s in X]

        # We cannot use the trainer.predict(), because it calls .eval() which internally disables dropout again
        with torch.no_grad():
            results = []

            for X_batch in more_itertools.chunked(X, 64):
                tokenized_stuff = tokenizer(
                    X_batch, truncation=True, padding=True, is_split_into_words=True, return_offsets_mapping=True
                )

                aligned_indices = self._align_token_indices(tokenizer, tokenized_stuff)

                pt_inputs = {
                    k: torch.tensor(v).detach().to(self._device())
                    for k, v in tokenized_stuff.items()
                    if k != "offset_mapping"
                }
                output = self._model(**pt_inputs)
                logits = output.logits.cpu().numpy()

                predictions = softmax(logits, axis=1)

                assert len(predictions) == len(aligned_indices) == len(X_batch)

                for i in range(len(X_batch)):
                    current_alignment = aligned_indices[i]
                    cur_predictions = predictions[i]
                    try:
                        aligned_predictions = cur_predictions[current_alignment]
                    except Exception as e:
                        print(e)

                    # Check that the sentence and its predictions have the same length
                    assert len(X_batch[i]) == len(
                        aligned_predictions
                    ), f"Do you maybe have spaces in a token?\n{X_batch[i]}"

                    aligned_predictions = normalize(aligned_predictions, norm="l1", axis=1)

                    results.append(aligned_predictions)

            return ak.Array(results)

    def _align_token_indices(self, tokenizer, tokenized_stuff) -> ak.Array:
        # https://huggingface.co/transformers/custom_datasets.html?highlight=offset_mapping#token-classification-with-w-nut-emerging-entities
        # https://discuss.huggingface.co/t/predicting-with-token-classifier-on-data-with-no-gold-labels/9373
        aligned_indices = []
        for sentence_id, offsets in enumerate(tokenized_stuff.offset_mapping):
            current = []

            decoded_tokens = tokenizer.convert_ids_to_tokens(tokenized_stuff.input_ids[sentence_id])

            for idx, offset in enumerate(offsets):
                if self._is_roberta:
                    keep = decoded_tokens[idx][0] == "Ġ"
                else:
                    keep = offset != (0, 0) and offset[0] == 0

                if keep:
                    current.append(idx)

            aligned_indices.append(current)

        return ak.Array(aligned_indices)

    def label_encoder(self) -> LabelEncoder:
        assert self._model, "Model not set, train first"

        le = LabelEncoder()

        le.classes_ = np.array(self._build_label_list())
        return le

    @my_backoff()
    def _build_tokenizer(self):
        # add_prefix_space is True because roberta needs it
        tokenizer = AutoTokenizer.from_pretrained(self._model_name, use_fast=True, add_prefix_space=self._is_roberta)

        return tokenizer

    def _tokenize_and_align_labels(self, texts: RaggedStringArray, labels: RaggedStringArray):
        tokenizer = self._tokenizer

        tokenized_inputs = tokenizer(
            texts.tolist(),
            padding=True,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        aligned_labels = []
        for i, current_labels in enumerate(labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(self._label_to_id[current_labels[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            aligned_labels.append(label_ids)

        tokenized_inputs["labels"] = aligned_labels
        return tokenized_inputs

    @my_backoff()
    def _build_model(self) -> AutoModelForTokenClassification:
        config = self._build_config()

        model = AutoModelForTokenClassification.from_pretrained(
            self._model_name,
            config=config,
        )

        return model

    def _build_config(self) -> AutoConfig:
        config = AutoConfig.from_pretrained(
            self._model_name, num_labels=len(self._id_to_label), id2label=self._id_to_label, label2id=self._label_to_id
        )

        return config

    @staticmethod
    def _device() -> Device:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        return device

    def _compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        label_list = self._build_label_list()

        # Remove ignored index (special tokens)
        y_true = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        y_pred = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            p = seqeval.metrics.precision_score(y_true, y_pred)
            r = seqeval.metrics.recall_score(y_true, y_pred)
            f1 = seqeval.metrics.f1_score(y_true, y_pred)
            a = seqeval.metrics.accuracy_score(y_true, y_pred)

        return {
            "precision": p,
            "recall": r,
            "f1": f1,
            "accuracy": a,
        }

    def _build_label_list(self) -> List[str]:
        return list(sorted(self._label_to_id.keys()))

    def has_dropout(self) -> bool:
        return True

    def use_dropout(self, is_activated: bool):
        self._use_mc_dropout = is_activated

    def _activate_mc_dropout_if_needed(self):
        assert self.has_dropout()

        def apply_dropout(m):
            if type(m) == nn.Dropout:
                if self._use_mc_dropout:
                    m.train()
                else:
                    m.eval()

        self._model.apply(apply_dropout)


class SequenceTaggingTrainDataset(Dataset):
    def __init__(self, tokenized_texts, encoded_labels):
        self.tokenized_texts = tokenized_texts
        self.encoded_labels = encoded_labels

    def __getitem__(self, idx):
        item = {
            "input_ids": self.tokenized_texts[idx],
            "labels": self.encoded_labels[idx],
        }
        return item

    def __len__(self):
        return len(self.encoded_labels)


class SequenceTaggingEvalDataset(Dataset):
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __getitem__(self, idx):
        item = {
            "input_ids": self.tokenized_texts[idx],
        }
        return item

    def __len__(self):
        return len(self.tokenized_texts)
