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


class CurriculumSpotter(Detector):
    """Spotting Spurious Data with Neural Networks

    Hadi Amiri, Timothy A. Miller, Guergana Savova
    https://aclanthology.org/N18-1182.pdf
    """

    def __init__(self, verbose: bool = True, max_epochs: int = 24):
        self._verbose = verbose
        self._max_epochs = max_epochs

    def error_detector_kind(self) -> DetectorKind:
        return DetectorKind.SCORER

    def score(self, texts: StringArray, labels: StringArray, **kwargs) -> npt.NDArray[float]:
        """Apply curriculum learning and score the items by the perceived difficulty during
        the curriculum training.

        Args:
            texts: a (num_instances, ) string sequence containing the texts for each instance
            labels: a (num_instances, ) string sequence containing the noisy label for each instance
        Returns:
            scores: a (num_instances,) numpy array of bools containing the flags after using CS
        """

        # Model setup
        model = CurriculumSpotterTransformerTextClassifier(verbose=self._verbose, max_epochs=self._max_epochs)
        cl_spotter_callback = CurriculumSpotterDatasetCallback(model)
        model.add_callback("cl_spotter", cl_spotter_callback)

        model.fit(texts, labels)

        return cl_spotter_callback.scores


class CurriculumSpotterDataset(Dataset):
    # https://discuss.pytorch.org/t/solved-will-change-in-dataset-be-reflected-on-dataloader-automatically/10206

    def __init__(self, tokenized_texts: Dict, encoded_labels: List[int]):
        self._tokenized_texts = tokenized_texts
        self._encoded_labels = encoded_labels

        self._mapping: Dict[int, int] = {i: i for i in range(len(encoded_labels))}

    def __getitem__(self, idx: int):
        new_idx = self._mapping[idx]

        item = {
            key: torch.as_tensor(val[new_idx], dtype=torch.int64).clone().detach()
            for key, val in self._tokenized_texts.items()
        }
        item["labels"] = torch.as_tensor(self._encoded_labels[new_idx], dtype=torch.int64).clone().detach()
        return item

    def __len__(self):
        return len(self._mapping)

    def update_mapping(self, new_dataset_mask: np.ndarray):
        # ``new_dataset`` is array of bool saying whether that instance is part of the new dataset

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
        return torch.as_tensor(self._encoded_labels, dtype=torch.int64).clone().detach()

    @property
    def true_len(self):
        return len(self._encoded_labels)


class CurriculumSpotterTransformerTextClassifier(TransformerTextClassifier):
    def _build_dataset(self, X: np.ndarray, y: np.ndarray, train: bool) -> CurriculumSpotterDataset:
        tokenized_texts = self._tokenizer(X.tolist(), truncation=True, padding=True)

        if train:
            self._label_encoder = LabelEncoder()
            encoded_labels = self._label_encoder.fit_transform(y)
        else:
            assert self._label_encoder
            encoded_labels = self._label_encoder.transform(y)

        dataset = CurriculumSpotterDataset(tokenized_texts, encoded_labels)

        return dataset


class CurriculumSpotterDatasetCallback(TrainerCallback):
    """Callback that is called during certain events (starting/ending training, ...)"""

    def __init__(self, model: TransformerTextClassifier):
        super().__init__()

        self.scores: Optional[np.ndarray] = None
        self._model = model

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        train_dataloader: DataLoader = kwargs["train_dataloader"]
        dataset: CurriculumSpotterDataset = train_dataloader.dataset

        self.scores = np.zeros(dataset.true_len)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model: PreTrainedModel = kwargs["model"]
        train_dataloader: DataLoader = kwargs["train_dataloader"]
        dataset: CurriculumSpotterDataset = train_dataloader.dataset

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
        dataset: CurriculumSpotterDataset = train_dataloader.dataset

        # Compute the losses on the *whole* dataset
        predictions, losses = self._get_predictions_and_losses(model, dataset.X, dataset.y)

        # Compute lambda
        lambda_ = self._compute_lambda(dataset.y.cpu().detach().numpy(), predictions, losses)

        # Build new dataset
        easy_mask = self._sample_easy(lambda_, losses)
        hard_mask = self._sample_hard(lambda_, int(state.epoch) / int(state.num_train_epochs), losses)

        new_dataset_mask = easy_mask | hard_mask
        dataset.update_mapping(new_dataset_mask)

        self._update_stat(hard_mask, losses)

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

    def _compute_lambda(self, y_gold: np.ndarray, y_pred: np.ndarray, losses: np.ndarray) -> float:
        # lambda is the average loss of correctly classified instances
        correct_indices = y_gold == y_pred
        losses_for_correct_instances = losses[correct_indices]
        l = losses_for_correct_instances.mean()
        return float(l)

    def _sample_easy(self, lambda_: float, losses: np.ndarray) -> np.ndarray:
        # Easy instances are instances with have a  loss of `l` or less
        # Return a boolean array saying whether an instance belongs to the new easy dataset

        easy_mask = losses <= lambda_
        return easy_mask

    def _sample_hard(self, lambda_: float, delta: float, losses: np.ndarray) -> np.ndarray:
        # Hard instances are instances with have a  loss of greater than `l`
        # we sample the `delta` easiest percent of the hardest

        # Return a boolean array saying whether an instance belongs to the new hard dataset
        assert 0.0 <= delta <= 1.0

        n = len(losses)

        k = int(n * delta)
        assert 0 <= k <= n

        hard_mask = losses > lambda_

        hard_indices = []
        loss_indices_sorted = np.argsort(losses)

        # Go through the indices from lowest to highest loss and collect the first `k` hard instances
        for i in loss_indices_sorted:
            if hard_mask[i]:
                hard_indices.append(i)

            if len(hard_indices) >= k:
                break

        result_mask = np.zeros(n, dtype=bool)
        result_mask[hard_indices] = True

        return result_mask

    def _update_stat(self, hard_mask: np.ndarray, losses: np.ndarray):
        num_of_hard_instances = np.count_nonzero(hard_mask)
        new_scores = hard_mask * (losses + 1 / num_of_hard_instances)
        self.scores += new_scores
