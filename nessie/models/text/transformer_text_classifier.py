import tempfile
from typing import Dict, List, Optional

import more_itertools
import numpy as np
import numpy.typing as npt
import torch
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.types import Device
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    IntervalStrategy,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from nessie.config import BERT_BASE
from nessie.models.model import Callbackable, TextClassifier
from nessie.types import StringArray
from nessie.util import RANDOM_STATE, my_backoff


class TransformerTextClassifier(TextClassifier, Callbackable):
    @my_backoff()
    def __init__(self, verbose: bool = True, max_epochs: int = 24, batch_size: int = 16, model_name: str = BERT_BASE):
        assert max_epochs is not None
        assert batch_size is not None

        self._model: Optional[PreTrainedModel] = None
        self._config: Optional[AutoConfig] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._verbose = verbose
        self._max_epochs = max_epochs
        self._batch_size = batch_size

        self._callbacks: Dict[str, TrainerCallback] = {}

        self._model_name = model_name

        self._use_mc_dropout = False

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

    @my_backoff()
    def fit(self, X: StringArray, y: StringArray):
        dataset = self._build_dataset(X, y, train=True)
        model = self._build_model(len(self._label_encoder.classes_))
        self._model = model

        model.train()

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

            callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
            callbacks.extend(self._callbacks.values())

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                eval_dataset=dataset,
                callbacks=callbacks,
                compute_metrics=compute_metrics,
            )

            trainer.train()

    @my_backoff()
    def predict(self, X: StringArray) -> npt.NDArray[str]:
        assert self._model, "Model not set for predicting, train first"

        predictions = self._predict(X)

        predictions = np.argmax(predictions, axis=1)
        assert len(X) == len(predictions)

        labels = self._label_encoder.inverse_transform(predictions)

        return np.array(labels)

    @my_backoff()
    def score(self, X: StringArray) -> npt.NDArray[float]:
        assert self._model, "Model not set for predicting, train first"

        predictions = self._predict(X)

        probs = np.max(predictions, axis=1)
        assert len(X) == len(probs)

        return np.array(probs)

    @my_backoff()
    def predict_proba(self, X: StringArray) -> npt.NDArray[float]:
        assert self._model, "Model not set for predicting, train first"

        predictions = self._predict(X)

        assert (len(X), len(self._label_encoder.classes_)) == predictions.shape

        return np.array(predictions)

    def label_encoder(self) -> LabelEncoder:
        return self._label_encoder

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

    def _build_dataset(self, X: StringArray, y: StringArray, train: bool) -> "TextClassificationDataset":
        tokenized_texts = self._tokenizer(list(X), truncation=True, padding=True)

        if train:
            self._label_encoder = LabelEncoder()
            encoded_labels = self._label_encoder.fit_transform(y)
        else:
            assert self._label_encoder
            encoded_labels = self._label_encoder.transform(y)

        dataset = TextClassificationDataset(tokenized_texts, encoded_labels)

        return dataset

    @my_backoff()
    def _build_model(self, num_labels: int) -> PreTrainedModel:
        config = AutoConfig.from_pretrained(self._model_name, num_labels=num_labels, classifier_dropout=0.25)
        model = AutoModelForSequenceClassification.from_pretrained(self._model_name, config=config)

        return model.to(self._device())

    @staticmethod
    def _device() -> Device:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        return device

    @my_backoff()
    def _predict(self, X: np.ndarray) -> np.ndarray:
        self._model.eval()
        self._activate_mc_dropout_if_needed()

        # We cannot use the trainer.predict(), because it calls .eval() which internally disables dropout again
        result = []

        with torch.no_grad():

            for X_batch in more_itertools.chunked(X, 64):
                tokenized_texts = self._tokenizer(X_batch, return_tensors="pt", truncation=True, padding=True)
                # pt_inputs = {k: torch.tensor(v).to(self._device()) for k, v in tokenized_texts.items()}
                pt_inputs = {k: torch.as_tensor(v).detach().to(self._device()) for k, v in tokenized_texts.items()}

                output = self._model(**pt_inputs)

                logits = output.logits.cpu().numpy()

                result.append(logits)

        logits = np.vstack(result)
        assert len(logits) == len(X)
        return softmax(logits, axis=1)

    def _build_prediction_trainer(self, output_dir: str) -> Trainer:
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=64,
            seed=RANDOM_STATE,
            disable_tqdm=not self._verbose,
            report_to="none",
            log_level="info" if self._verbose else "error",
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
        )

        return trainer

    def add_callback(self, name: str, callback: TrainerCallback):
        self._callbacks[name] = callback


class TextClassificationDataset(Dataset):
    def __init__(self, tokenized_texts: Dict, encoded_labels: List[int]):
        self.tokenized_texts = tokenized_texts
        self.encoded_labels = encoded_labels

    def __getitem__(self, idx):
        item = {
            key: torch.as_tensor(val[idx], dtype=torch.int64).clone().detach()
            for key, val in self.tokenized_texts.items()
        }
        item["labels"] = torch.as_tensor(self.encoded_labels[idx], dtype=torch.int64).clone().detach()
        return item

    def __len__(self):
        return len(self.encoded_labels)


class TextClassificationEvalDataset(Dataset):
    def __init__(self, tokenized_texts: Dict):
        self.tokenized_texts = tokenized_texts

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}

    def __len__(self):
        return len(self.tokenized_texts["input_ids"])


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
    }
