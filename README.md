# nessie

## Installation

    pip install nessie

This installs the package with default dependencies and PyTorch with only CPU support.
If you want to use your own PyTorch version, you need to force install it afterwards. 
If you need `faiss-gpu`, then you should also install that manually afterwards.

## Development

    poetry install

In order to install your own PyTorch with CUDA, you can run

    poetry shell
    poe force-cuda113

or install it manually in the poetry environment.