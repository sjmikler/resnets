# ResNets in Tensorflow 2.0 on CIFAR-10

Nice and tidy implementation of various neural networks for classification in tensorflow 2.0. \
Everything in one place with results matching those reported in papers.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gahaalt/cifar-vs-tensorflow2/blob/master/Playground.ipynb)

Requirements:
- tensorflow 2.0
- tensorflow_datasets
- tensorboard

## Implemented models (cifar versions only):
```
★ if pretrained weights are available and ☆ if not

From "Very Deep Convolutional Network for Large-Scale Image Recognition",
     "On Correlation of Features Extracted by Deep NeuralNetworks":
    ★ VGG11
    ★ VGG13
    ★ VGG16
    ★ VGG19

From "Deep Residual Learning for Image Recognition":
    ★ ResNet20
    ★ ResNet32
    ★ ResNet44
    ★ ResNet56
    ☆ ResNet110

From "Identity Mappings in Deep Residual Networks" (with preactivated layers):
    ★ ResNet110
    ★ ResNet164
    ★ ResNet1001
    
From "Wide Residual Networks":
    ☆ Wide ResNet-16-4
    ☆ Wide ResNet-40-4
    ☆ Wide ResNet-16-8
    ☆ Wide ResNet-28-10
    
Incoming in the near future:
    ☆ ResNeXt
    
★ if pretrained weights are available and ☆ if not
```

## How to get keras models:
```
# ResNet110 v1
from Models.Resnets import cifar_resnet110
model = cifar_resnet20('original', shortcut_type='A')

# ResNet110 v2 pretrained
from Models.Resnets import cifar_resnet110
model = cifar_resnet110('preactivated', load_weights=True)

# ResNet164 pretrained
from Models.Resnets import cifar_resnet164
model = cifar_resnet164(load_weights=True)
```
They are ready to train or test with 'fit' method.

## How to train:
- Set experiments in experiments.yaml
- Run using: python run_experiments.py
- Open logs with tensorboard

### Default training schedule:
Works well for ResNets v1 and v2
- SGD with momentum **0.9**
- for iterations **[0, 400)** LR = **0.01** (warm-up)
- for iterations **[400, 32000)** LR = **0.1**
- for iterations **[32000, 48000)** LR = **0.01**
- for iterations **[48000, 64000)** LR = **0.001**
- L2 regularization = **0.0001**

### Training ResNet110 v1 example:
```
module: 'Models.Resnets'      # .py file with models (required)
model: 'cifar_resnet110'      # function that creates model (required)
model_parameters:
    block_type: 'original'    # original for Resnet v1, preactivated for Resnet v2
    shortcut_mode: 'A'        # A or B as in Deep Residual Learning for Image Recognition
train_parameters:
    logdir: 'logs'            # folder for tensorboard (required)
    run_name: 'resnet110_v1'  # name of the run in tensorboard (required)
    log_interval: 400         # how often statistics are printed and saved to tensorboard
    val_interval: 4000        # how often validation on the test set is done
skip_error_test: False        # whether to skip a quick run before beginning the actual training
```
Note that not all parameters are required.

### Training Wide ResNet-40-4 example:
```
module: 'Models.Resnets'
model: 'WRN_40_4'
model_parameters: {}
train_parameters:
    logdir: 'logs'
    run_name: 'WRN_40_4'
    log_interval: 400
    val_interval: 4000
    lr_values: [0.1, 0.02, 0.004, 0.0008]
    lr_boundaries: [24000, 48000, 64000, 80000]
    nesterov: True
```

## Comparision with results reported on CIFAR-10:

#### VGG Networks
| architecture | parameters | reported best | this repository |
| ---: | :---: | :---: | :---: |
| VGG11 | 9.2M | 7.81 | **7.98** |
| VGG13 | 9.4M | 6.35 | **6.17** |
| VGG16 | 14.7M | 6.49 | **6.34** |
| VGG19 | 20.0M | 6.76 | **6.72** |

#### ResNets v1
| architecture | parameters | reported best | this repository |
| ---: | :---: | :---: | :---: |
| ResNet20 | 0.27M | 8.75 | **7.99** |
| ResNet32 | 0.46M | 7.51 | **7.40** |
| ResNet44 | 0.66M | 7.17 | **6.83** |
| ResNet56 | 0.85M | 6.97 | **6.23** |
| ResNet110 | 1.7M | 6.43 | 6.26 |

#### ResNets v2
| architecture | parameters | reported mean | this repository |
| ---: | :---: | :---: | :---: |
| ResNet110 | 1.7M | 6.37 | **5.98** |
| ResNet164 | 1.7M | 5.46 | **5.27** |
| ResNet1001 | 10.3M | 4.92 | **5.06** |

#### Wide ResNets
| architecture | parameters | reported mean | this repository |
| ---: | :---: | :---: | :---: |
| WRN-16-4 | 2.7M | 5.02 | ? |
| WRN-40-4 | 8.9M | 4.53 | 4.46 |
| WRN-16-8 | 11.0M | 4.27 | ? |
| WRN-28-10 | 36.5M | 4.00 | ? |
| + dropout | 36.5M | 3.89 | ? |

### Training curves
All training logs are available in saved_logs folder. You can open it with tensorboard and compare them with yours.

### Differences:
I did my best to make the implementation identical to the original one, however there are subtle differences in the training:

- I use bigger L2 regularization for ResNets - 1e-4 instead of 5e-5
- All networks are trained on 50.000 examples, whereas some papers use only 45.000 examples
- I use warm-up iterations for all the networks, not only for ResNet-110 as in original paper

With this repository you can easily define and train the networks for other datasets, e.g. ImageNet.
