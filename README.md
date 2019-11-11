# CIFAR vs Tensorflow 2.0

Nice and tidy implementation of various neural networks for classification in tensorflow 2.0. \
Everything in one place with results matching those reported in papers.


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gahaalt/cifar-vs-tensorflow2/blob/master/Playground.ipynb)

Requirements:
- tensorflow-gpu=2.0
- tensorflow_datasets
- tensorboard

## Implemented models (cifar versions only):
```
From "Very Deep Convolutional Neural Network Based Image Classification Using Small Training Sample Size":
    - VGG11
    - VGG13
    - VGG16
    - VGG19

From "Deep Residual Learning for Image Recognition":
    - ResNet20 (1)
    - ResNet32 (1)
    - ResNet44 (1)
    - ResNet56
    - ResNet110

From "Identity Mappings in Deep Residual Networks" (with preactivated layers):
    - ResNet20 (1)
    - ResNet32
    - ResNet44
    - ResNet56
    - ResNet110 (1)
    - ResNet164
    - ResNet1001
    
From "Wide Residual Networks":
    - Wide ResNet-40-4
    - Wide ResNet-16-8
    - Wide ResNet-28-10
    
Incoming:
    - ResNeXt
    
(1) - weights can be loaded with 'load_weights' parameter
```

## How to get keras model:
```
# ResNet110 v1
from Models.Resnets import cifar_resnet110
model = cifar_resnet20(block_type='original', load_weights=True)

# ResNet110 v2
from Models.Resnets import cifar_resnet110
model = cifar_resnet110(block_type='preactivated', load_weights=True)

# ResNet164
from Models.Resnets import cifar_resnet164
model = cifar_resnet164(load_weights=True)
```

## How to train:
- Set experiments in experiments.yaml
- Run using: python run_experiments.py
- Open logs with tensorboard

## Default training schedule:
- SGD with momentum **0.9**
- warm-up LR = **0.01** for iterations **[0, 400)**
- LR = **0.1** for iterations **[400, 32000)**
- LR = **0.01** for iterations **[32000, 48000)**
- LR = **0.001** for iterations **[48000, 64000)**
- Weight decay = **0.0001**

## Example of an experiment:
```
module: 'Models.Resnets'      # .py file with models (required)
model: 'cifar_resnet110'      # function that creates model (required)
model_parameters:
    shortcut_mode: 'B'        # A or B as in Deep Residual Learning for Image Recognition
    block_type: 'original'    # original for Resnet v1, preactivated for Resnet v2
train_parameters:
    logdir: 'logs'            # folder for tensorboard (required)
    run_name: 'resnet110_v1'  # name of the run in tensorboard (required)
    num_steps: 64000          # iterations after which the training ends
    log_interval: 400         # how often statistics are printed and saved to tensorboard
    val_interval: 4000        # how often validation on the test set is done
skip_error_test: False        # whether to skip a quick run before beginning the actual training
```


## Error rate comparision with results reported on CIFAR-10:

| architecture | parameters | reported best | reported mean | this repository |
| ---: | :---: | :---: | :---: | :---: |
| VGG16 | | 8.45 | ? | 7.26 |
| ResNet20 v1 | 0.27M | 8.75 | ? | 8.39-8.55 |
| ResNet32 v1 | 0.46M | 7.51 | ? | 7.46 |
| ResNet44 v1 | 0.66M | 7.17 | ? | 7.08 |
| ResNet56 v1 | 0.85M | 6.97 | ? | ? |
| ResNet110 v1 | 1.7M | 6.43 | 6.61 | ? |
| ResNet164 v1 | 1.7M | ? | 5.93 | ? |
| ResNet20 v2 | 0.27M | ? | ? | 7.92 |
| ResNet110 v2 | 1.7M | ? | 6.37 | 5.89-6.1 |
| ResNet164 v2 | 1.7M | ? | 5.46 | ? |
| ResNet1001 v2 | 10M | ? | 4.92 | ? |
| WRN-40-4 | 8.9M | ? | 4.53 | ? |
| WRN-16-8 | 11.0M | ? | 4.27 | ? |
| WRN-28-10 | 36.5M | ? | 4.00 | ? |
