# GAN-NAML-Project

# JAX-based Generative Adversarial Network (GAN) on MNIST

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Usage](#usage)
- [Results](#results)
- [Saving and Loading Model](#saving-and-loading-model)
- [Contributing](#contributing)

## Introduction

This project implements a **Generative Adversarial Network (GAN)** using [JAX](https://jax.readthedocs.io/en/latest/), a high-performance machine learning library. The model is trained on the **MNIST dataset**. The GAN consists of two neural networks:
- A **Generator** that creates synthetic images.
- A **Discriminator** that distinguishes between real and generated images.

## Requirements

The project uses the following Python libraries:

- `jax` for numerical computations
- `flax` for neural network modules
- `optax` for optimization
- `tensorflow_datasets` for loading the MNIST dataset
- `numpy` and `jax.numpy` for array manipulations
- `matplotlib` for visualization

You can install the dependencies using:

```bash
pip install jax jaxlib flax optax tensorflow-datasets matplotlib
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/gan-jax.git
   cd gan-jax
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The MNIST dataset is loaded using `tensorflow_datasets`. The dataset contains 28x28 grayscale images of handwritten digits. We use only images of digits **1**, **4**, and **8**, and preprocess them by normalizing and scaling to the range [-1, 1].

## Model Architecture

### Generator
The **Generator** takes a random latent vector as input and outputs a 28x28 grayscale image. The architecture includes:
- Dense layers with ReLU activations
- A final dense layer with a `tanh` activation to produce image pixel values in the range [-1, 1]

### Discriminator
The **Discriminator** takes an image as input and outputs a single value indicating whether the image is real or generated. The architecture includes:
- Dense layers with Leaky ReLU activations
- A final dense layer with a `sigmoid` activation to output a probability

## Training

The GAN is trained using the following loss functions:
- **Generator Loss**: Encourages the generator to produce images that the discriminator classifies as real.
- **Discriminator Loss**: Encourages the discriminator to correctly classify real and fake images.

The training loop consists of:
1. Sampling a batch of real images from the dataset.
2. Generating fake images using the generator.
3. Computing the discriminator and generator losses.
4. Updating the model parameters using `optax.adam`.

To train the model, simply run:

```bash
python main.py
```

The training process is set to run for **10,000 epochs**, with losses printed every 100 epochs.

## Usage

You can visualize the generated images during training using the `show_digits` function. After training, you can generate new images with:

```python
key = jax.random.PRNGKey(0)
latent_dim = 64
fake_images = gen.apply(gen_params, jax.random.normal(key, (10, latent_dim)))
show_digits(fake_images, n=10, rows=1)
```

## Results

During training, the generator gradually learns to produce realistic images of digits 1, 4, and 8. Sample outputs are saved and displayed every 1,000 epochs. Below is an example of generated images after training:

- **Generated images at epoch 1000**
- **Generated images at epoch 5000**
- **Generated images at epoch 10000**

## Saving and Loading Model

The generator parameters can be saved to a file using:

```python
import pickle

with open("gen_params.pkl", "wb") as f:
    pickle.dump(gen_params, f)
```

To load the saved generator parameters:

```python
with open("gen_params.pkl", "rb") as f:
    gen_params = pickle.load(f)
```

## Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Create a new Pull Request.
