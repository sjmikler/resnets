# cifar-vs-tensorflow2
Nice and tidy implementation of classical neural networks for classification in tensorflow 2.0

Requirements:
- tensorflow-gpu=2.0
- tensorflow_datasets

Set experiments in experiments.yaml \
Run using: python run_experiments.py

## Implemented models (cifar versions only):
```
  From "Very Deep Convolutional Neural Network Based Image Classification Using Small Training Sample Size":
    - VGG11
    - VGG16
    - VGG19

  From "Deep Residual Learning for Image Recognition":
    - ResNet20
    - ResNet32
    - ResNet44
    - ResNet56
    - ResNet110
    - ResNet1001

  From "Identity Mappings in Deep Residual Networks" (with preactivated layers):
    - ResNet20
    - ResNet32
    - ResNet44
    - ResNet56
    - ResNet110
    - ResNet164
    - ResNet1001
    
  TODO:
    - ResNeXt
    - EfficientNet
```

## Default training schedule:
- SGD with momentum **0.9**
- warm-up LR = **0.01** for iterations **[0, 400)**
- LR = **0.1** for iterations **[400, 32000)**
- LR = **0.01** for iterations **[32000, 48000)**
- LR = **0.001** for iterations **[48000, 64000)**
- Weight decay = **0.0001**

## Example of an experiment:
```
module: 'Models.Resnets'
model: 'cifar_resnet110'
model_parameters:
    shortcut_mode: 'B'        # A or B as in Deep Residual Learning for Image Recognition
    block_type: 'original'    # original for Resnet v1, preactivated for Resnet v2
train_parameters:
    logdir: 'logs'
    run_name: 'resnet110_v1'
    num_steps: 64000          # iterations after which the training ends
    log_interval: 400         # how often statistics are printed and saved to tensorboard
    val_interval: 4000        # how often validation on the test set is done
skip_error_test: True         # whether to skip a quick run before beginning the actual training
```


## Error rate comparision with results reported on CIFAR-10:

| architecture | parameters | reported best | reported mean | this repository |
| ---: | :---: | :---: | :---: | :---: |
| VGG16 | | 8.45 | ? | 7.26 |
| ResNet20 v1 | 0.27M | 8.75 | ? | 8.39 |
| ResNet32 v1 | 0.46M | 7.51 | ? | ? |
| ResNet44 v1 | 0.66M | 7.17 | ? | ? |
| ResNet56 v1 | 0.85M | 6.97 | ? | ? |
| ResNet110 v1 | 1.7M | 6.43 | 6.61 | ? |
| ResNet164 v1 | 1.7M | ? | 5.93 | ? |
| ResNet110 v2 | 1.7M | ? | 6.37 | 6.1 |
| ResNet164 v2 | 1.7M | ? | 5.46 | ? |
| ResNet1001 v2 | | ? | 4.92 | ? |
