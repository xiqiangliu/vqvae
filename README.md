# Vector Quantized Variational Autoencoder

This is an implementation of [Vector Quantized Variational Autoencoder](https://arxiv.org/abs/1711.00937) for the final project of ECE 176: Introduction to Deep Learning & Applications at UC San Diego.

## Environment Setup

First, ensure you have a clean Python distribution that has version `>=3.11`. One way of doing it is through miniconda:

```bash
conda create --name vqvae python=3.12
conda activate vqvae
```

Then, you can install poetry:

```bash
pip install poetry
```

Finally, install the dependencies and the package itself:

```bash
poetry install
```

To download the CelebA dataset, you might need the library `gdown`. You can install it with:

```bash
poetry install --with download
```

when installing the package.


## Training

To train the model, you can run (assuming you're in the root of the repository):

```bash
python scripts/train/cifar10.py
python scripts/train/celeba.py
```

The hyperparameters can be changed directly in the script files. The results will be saved in the `logs` directory.
