# Resnets

Tidy implementation of classical neural networks for classification.

Available frameworks:

* Jax
* PyTorch
* TensorFlow

Everything in one place with results matching those reported in papers.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gahaalt/cifar-vs-tensorflow2/blob/master/Playground.ipynb)

## Installation

Install the package with pip:

```bash
pip install resnets
```

## Available architectures

### VGG

| architecture | parameters | reported best | this repository | CIFAR-10 ckpt | ImageNet ckpt |
|-------------:|:----------:|:-------------:|:---------------:|---------------|---------------|
|        VGG11 |    9.2M    |     7.81      |      7.98       |               |               |
|        VGG13 |    9.4M    |     6.35      |      6.17       |               |               |
|        VGG16 |   14.7M    |     6.49      |      6.34       |               |               |
|        VGG19 |   20.0M    |     6.76      |      6.72       |               |               |

Sources:

* [Very Deep Convolutional Network for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
* [On Correlation of Features Extracted by Deep Neural Networks](https://arxiv.org/abs/1901.10900)

### ResNet

| architecture | parameters | reported best | this repository | CIFAR-10 ckpt | ImageNet ckpt |
|-------------:|:----------:|:-------------:|:---------------:|---------------|---------------|
|    ResNet-20 |   0.27M    |     8.75      |      7.99       |               |               |
|    ResNet-32 |   0.46M    |     7.51      |      7.40       |               |               |
|    ResNet-44 |   0.66M    |     7.17      |      6.83       |               |               |
|    ResNet-56 |   0.85M    |     6.97      |      6.23       |               |               |
|   ResNet-110 |    1.7M    |     6.37      |      5.98       |               |               |
|   ResNet-164 |    1.7M    |     5.46      |      5.27       |               |               |
|  ResNet-1001 |   10.3M    |     4.92      |      5.06       |               |               |

Sources:

* [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
* [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)

### Wide ResNet

| architecture | parameters | reported mean | this repository | CIFAR-10 ckpt | ImageNet ckpt |
|-------------:|:----------:|:-------------:|:---------------:|---------------|---------------|
|     WRN-16-4 |    2.7M    |     5.02      |                 |               |               |
|     WRN-40-4 |    8.9M    |     4.53      |      4.46       |               |               |
|     WRN-16-8 |   11.0M    |     4.27      |                 |               |               |
|    WRN-28-10 |   36.5M    |     4.00      |                 |               |               |
|   WRN-28-10+ |   36.5M    |     3.89      |                 |               |               |

(+ with dropout)

Sources:

* [Wide Residual Networks](https://arxiv.org/abs/1605.07146)

### ResNeXt