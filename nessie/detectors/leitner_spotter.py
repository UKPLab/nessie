import logging
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import (
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from nessie.detectors.error_detector import Detector, DetectorKind
from nessie.models.text.transformer_text_classifier import (
    TextClassificationDataset,
    TransformerTextClassifier,
)
from nessie.types import StringArray
from nessie.util import RANDOM_STATE

Queues = List[List[int]]


class LeitnerSpotter(Detector):
    """Spotting Spurious Data with Neural Networks

    Hadi Amiri, Timothy A. Miller, Guergana Savova
    https://aclanthology.org/N18-1182.pdf
    """

    def __init__(self, verbose: bool = True, max_epochs: int = 48):
        self._verbose = verbose
        self._max_epochs = max_epochs

    def error_detector_kind(self) -> DetectorKind:
        return DetectorKind.SCORER

    def score(self, texts: StringArray, labels: StringArray, **kwargs) -> npt.NDArray[float]:
        """Apply curriculum learning via Zettelkasten and score the items by the perceived difficulty during
        the curriculum training.

        Args:
            texts: a (num_instances, ) string sequence containing the texts for each instance
            labels: a (num_instances, ) string sequence containing the noisy label for each instance
        Returns:
            scores: a (num_instances,) numpy array of bools containing the flags after using CS
        """

        model = LeitnerSpotterTransformerTextClassifier(verbose=self._verbose, max_epochs=self._max_epochs)
        cl_spotter_callback = LeitnerSpotterDatasetCallback(model)
        model.add_callback("cl_spotter", cl_spotter_callback)

        model.fit(texts, labels)

        return cl_spotter_callback.scores


class LeitnerSpotterDataset(Dataset):
    # https://discuss.pytorch.org/t/solved-will-change-in-dataset-be-reflected-on-dataloader-automatically/10206

    def __init__(self, tokenized_texts: Dict, encoded_labels: List[int]):
        self._tokenized_texts = tokenized_texts
        self._encoded_labels = encoded_labels

        self._mapping: Dict[int, int] = {i: i for i in range(len(encoded_labels))}

    def __getitem__(self, idx: int):
        new_idx = self._mapping[idx]

        item = {key: torch.tensor(val[new_idx], dtype=torch.int64) for key, val in self._tokenized_texts.items()}
        item["labels"] = torch.tensor(self._encoded_labels[new_idx], dtype=torch.int64)
        return item

    def __len__(self):
        return len(self._mapping)

    def update_mapping(self, new_dataset_mask: np.ndarray):
        # `new_dataset` is array of bool saying whether that instance is part of the new dataset

        # We build a mapping that maps instances from [0, |new_dataset|]
        # to indices in the original dataset
        count = 0
        mapping = {}
        for idx, e in enumerate(new_dataset_mask):
            # Instance is part of new dataset
            if e:
                mapping[count] = idx

                count += 1

        self._mapping = mapping

    @property
    def X(self):
        return self._tokenized_texts

    @property
    def y(self):
        return torch.tensor(self._encoded_labels, dtype=torch.int64)

    @property
    def true_len(self):
        return len(self._encoded_labels)


class LeitnerSpotterTransformerTextClassifier(TransformerTextClassifier):
    def _build_dataset(self, X: np.ndarray, y: np.ndarray, train: bool) -> "LeitnerSpotterDataset":
        tokenized_texts = self._tokenizer(X.tolist(), truncation=True, padding=True)

        if train:
            self._label_encoder = LabelEncoder()
            encoded_labels = self._label_encoder.fit_transform(y)
        else:
            assert self._label_encoder
            encoded_labels = self._label_encoder.transform(y)

        dataset = LeitnerSpotterDataset(tokenized_texts, encoded_labels)

        return dataset


class LeitnerSpotterDatasetCallback(TrainerCallback):
    def __init__(self, model: TransformerTextClassifier, number_of_queues: int = 5):
        super().__init__()

        self._number_of_queues = number_of_queues
        self._model = model

        self.scores: Optional[np.ndarray] = None
        self._queues: Optional[Queues] = None
        self._training_mask: Optional[np.ndarray] = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        train_dataloader: DataLoader = kwargs["train_dataloader"]
        dataset: LeitnerSpotterDataset = train_dataloader.dataset
        self._queues = [[] for _ in range(self._number_of_queues)]

        # In the first batch, all instances are in the lowest queue
        self._queues[0] = list(range(dataset.true_len))

        # In the first epoch, we train on the complete dataset
        self._training_mask = np.ones(dataset.true_len, dtype=bool)

        self.scores = np.zeros(dataset.true_len)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model: PreTrainedModel = kwargs["model"]
        train_dataloader: DataLoader = kwargs["train_dataloader"]
        dataset: LeitnerSpotterDataset = train_dataloader.dataset

        # Break ties for instances that have never been classified as hard by using the loss instead
        _, losses = self._get_predictions_and_losses(model, dataset.X, dataset.y)
        self.scores += (self.scores == 0) * losses

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs["model"]
        train_dataloader: DataLoader = kwargs["train_dataloader"]
        logging.info(f"Dataset size is [{len(train_dataloader.dataset)}]")

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model: PreTrainedModel = kwargs["model"]
        train_dataloader: DataLoader = kwargs["train_dataloader"]
        dataset: LeitnerSpotterDataset = train_dataloader.dataset

        # Compute the losses on the *whole* dataset
        y_pred, losses = self._get_predictions_and_losses(model, dataset.X, dataset.y)

        y_gold = dataset.y.cpu().detach().numpy()

        # Promote and demote
        self._queues = self._compute_new_queues(y_gold, y_pred)

        # Build new dataset mask
        self._training_mask = self._build_training_mask(int(state.epoch))

        dataset.update_mapping(self._training_mask)

        self._update_stat(losses)

    def _get_predictions_and_losses(self, model: PreTrainedModel, X, y: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        with tempfile.TemporaryDirectory() as tmpdirname:
            training_args = TrainingArguments(
                output_dir=tmpdirname,
                per_device_eval_batch_size=64,
                seed=RANDOM_STATE,
                disable_tqdm=True,
                report_to="none",
            )

            trainer = Trainer(
                model=model,
                args=training_args,
            )

            full_dataset = TextClassificationDataset(X, y)

            output = trainer.predict(full_dataset)

            # Compute predictions
            probas = softmax(output.predictions, axis=1)
            predictions = np.argmax(probas, axis=1)

            # Compute losses
            logits = torch.from_numpy(output.predictions)
            loss_fct = CrossEntropyLoss(reduction="none")
            losses = loss_fct(logits.view(-1, model.num_labels), y.view(-1))
            losses = losses.cpu().detach().numpy()

            assert len(predictions) == len(losses)

            return predictions, losses

    def _compute_new_queues(self, y_gold: np.ndarray, y_pred: np.ndarray) -> Queues:
        assert len(y_gold) == len(y_pred)

        correct_indices = y_gold == y_pred

        new_queues = [[] for _ in range(self._number_of_queues)]

        for q, queue in enumerate(self._queues):
            for idx in queue:
                # If the instance was not part of the training this epoch, we just skip
                if not self._training_mask[idx]:
                    continue

                # If the item was correctly classified, we promote it,
                # otherwise, we demote it to queue 0
                if correct_indices[idx]:
                    new_idx = min(self._number_of_queues, idx + 1)
                else:
                    new_idx = 0

                new_queues[q].append(new_idx)

        return new_queues

    def _build_training_mask(self, epoch: int) -> np.ndarray:
        new_mask = np.zeros_like(self._training_mask)

        for q, queue in enumerate(self._queues):
            if epoch % (2**q) != 0:
                # The time for this queue has not come yet
                continue

            for idx in queue:
                new_mask[idx] = True

        return new_mask

    def _update_stat(self, losses: np.ndarray):
        num_of_q0_instances = len(self._queues[0])

        for idx in self._queues[0]:
            self.scores[idx] += losses[idx] + 1 / num_of_q0_instances
