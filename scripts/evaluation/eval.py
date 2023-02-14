# library imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchsummary import summary
import json
import random
import numpy as np

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

from PIL import Image
Image.MAX_IMAGE_PIXELS = 120000000

import logging
logging.basicConfig(level=logging.INFO)
from sklearn.metrics import classification_report

# torch parameters being used
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
logging.info(f'Torch Version: {TORCH_VERSION}, Cuda Version: {CUDA_VERSION}')

# local imports
from test_helper_functions import create_datasets, create_dataloaders, test_model

# read in config file
with open('test_config.json') as f:
    config = json.load(f)
    logging.info('Test Config File Read Successfully.')

# create datasets and dataloaders
train_dataset, val_dataset, test_dataset, class_names, num_classes = create_datasets(config["data_dir"], config["train_perc"], config["val_perc"], config["test_perc"])
logging.info('Train, Validation and Test Datasets Created Successfully.')

dataloaders, dataset_sizes = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=config["batch_size"], num_workers = config["num_workers"])
logging.info('Train, Validation and Test Dataloaders Created Successfully.')

net = torch.load(config["model_dir"])
logging.info('Model Loaded Successfully.')

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f'Device {device} Being Used.')

all_preds, all_labels = test_model(net, dataloaders, device)
print(classification_report(all_labels, all_preds, target_names=class_names))


# resnet18_full.pt
# 
#                 precision    recall  f1-score   support
# 
#      neutral       0.91      0.95      0.93      8471
#         nsfw       0.86      0.74      0.80      2288
#     violence       0.74      0.59      0.65       617
# 
#     accuracy                           0.89     11376
#     macro avg      0.83      0.76      0.79     11376
#     weighted avg   0.89      0.89      0.89     11376


# resnet_152_only_fc_orig.pt
# 
#               precision    recall  f1-score   support
# 
#      neutral       0.94      0.93      0.93      8471
#         nsfw       0.80      0.85      0.82      2288
#     violence       0.74      0.68      0.71       617
# 
#     accuracy                           0.90     11376
#     macro avg      0.83      0.82      0.82     11376
#     weighted avg   0.90      0.90      0.90     11376


# vgg19_bn_only_fc.pt
# 
#                 precision    recall  f1-score   support
# 
#     neutral       0.91      0.95      0.93      8471
#     nsfw          0.85      0.77      0.81      2288
#     violence      0.71      0.59      0.65       617
# 
#     accuracy                          0.89     11376
#     macro avg     0.82      0.77      0.80     11376
#     weighted avg  0.89      0.89      0.89     11376 