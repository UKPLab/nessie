import logging
import os
import random
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import List

import backoff
import datasets
import numpy as np
import pandas
import requests
import wget as wget

RANDOM_STATE = 42


def setup_logging():
    import transformers

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # only log really bad events
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    datasets.logging.set_verbosity_error()

    transformers.logging.set_verbosity_error()

    # Backoff by default does not log, so we add it
    logging.getLogger("backoff").addHandler(logging.StreamHandler())


def get_logger(level=logging.DEBUG, filename=None):
    log = logging.getLogger(__name__)
    log.setLevel(level)

    if log.handlers:
        return log

    formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%m/%d/%Y %H:%M:%S")

    if filename is not None:
        file_handler = logging.FileHandler(str(filename))
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log


def download_file(url: str, target_path: Path) -> bool:
    logging.info("Downloading: [%s]", str(target_path.resolve()))
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        logging.info("File already exists: [%s]", str(target_path.resolve()))
        return False

    wget.download(url, str(target_path.resolve()))
    return True


@contextmanager
def tempinput(data):
    temp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
    temp.write(data)
    temp.close()
    try:
        yield Path(temp.name)
    finally:
        os.unlink(temp.name)


def write_sentence_classification_csv(p: Path, texts: List[str], gold_labels: List[str], noisy_labels: List[str]):
    p.parent.mkdir(exist_ok=True, parents=True)

    data = {"texts": texts, "gold": gold_labels, "noisy": noisy_labels}

    df = pandas.DataFrame(data)

    df.to_csv(p, index=False, sep="\t", header=False)


def set_my_seed(seed: int):
    import pyro
    import torch
    import transformers

    global RANDOM_STATE
    RANDOM_STATE = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    pyro.set_rng_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


my_backoff = lambda: backoff.on_exception(
    backoff.expo, (requests.exceptions.RequestException, EnvironmentError), max_tries=13
)
