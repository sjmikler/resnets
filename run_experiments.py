import os
import yaml
import logging
import importlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
logging.getLogger('tensorflow').disabled = True
from cifar_training_tools import cifar_training, cifar_error_test


def print_dict(d, tabs=0):
    tab = '\t'
    for key in d:
        if type(d[key]) == dict:
            print(f"{tab*tabs}{key}:")
            print_dict(d[key], tabs+1)
        else:
            print(f"{tab*tabs}{key}: {d[key]}")

            
print('\n' + '#' * 19)
print("TESTING FOR ERRORS!")
print('#' * 19)

stream = open('experiments.yaml', 'r')
for exp in yaml.safe_load_all(stream):
    if 'skip_error_test' in exp and exp['skip_error_test']:
        continue
        
    model = getattr(importlib.import_module(exp['module']), exp['model'])
    cifar_error_test(model(**exp['model_parameters']))
    print("OK!")
    

print('\n' + '#' * 22)
print("MODEL TRAINING BEGINS!")
print('#' * 22)

stream = open('experiments.yaml', 'r')
for exp in yaml.safe_load_all(stream):
    print(); print_dict(exp); print();
    
    model = getattr(importlib.import_module(exp['module']), exp['model'])
    cifar_training(model(**exp['model_parameters']), **exp['train_parameters'])