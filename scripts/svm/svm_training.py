# library imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchsummary import summary
import json
import random
import numpy as np
from sklearn.metrics import classification_report

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

from PIL import Image
Image.MAX_IMAGE_PIXELS = 120000000

import logging
logging.basicConfig(level=logging.INFO)

# local imports
from svm_helper_functions import create_datasets, create_dataloaders, train_model, test_model

# torch parameters being used
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
logging.info(f'Torch Version: {TORCH_VERSION}, Cuda Version: {CUDA_VERSION}')

class SVM_Loss(torch.nn.modules.Module):
    """
    SVM Loss function
    """    
    def __init__(self):
        """
        Initialize the SVM Loss function
        """
        super(SVM_Loss,self).__init__()

    def forward(self, outputs, labels, batch_size):
        """
        Forward pass of the SVM Loss function
        """
        return torch.sum(torch.clamp(1 - outputs.t()*labels, min=0))/batch_size

def run_script():
    """
    Driving function for the script. Trains a model on the training set while evaluating it on the validation set.
    Saves the best model 

    """
    # read in config file
    with open('svm_config.json') as f:
        config = json.load(f)
        logging.info('SVM Config File Read Successfully.')

    # create datasets and dataloaders
    train_dataset, val_dataset, test_dataset, class_names, num_classes = create_datasets(config["data_dir"], config["train_perc"], config["val_perc"], config["test_perc"])
    logging.info('Train, Validation and Test Datasets Created Successfully.')

    dataloaders, dataset_sizes = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=config["batch_size"], num_workers = config["num_workers"])
    logging.info('Train, Validation and Test Dataloaders Created Successfully.')

    # Parameters for training
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    input_size = config["input_size"]
    learning_rate = config["learning_rate"]
    momentum = config["momentum"]

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Device {device} Being Used.')

    # SVM regression model and Loss
    svm_model = nn.Linear(input_size,num_classes)

    # Loss and optimizer
    svm_loss_criteria = SVM_Loss()
    svm_optimizer = torch.optim.SGD(svm_model.parameters(), lr=learning_rate, momentum=momentum)
    total_step = len(dataloaders["train"])

    model = train_model(svm_model, input_size, svm_loss_criteria, svm_optimizer, dataloaders, batch_size, device, num_epochs)

    # save model
    torch.save(model, config["model_dir"])
    logging.info('Model Saved Successfully.')

    # test the model
    all_preds, all_labels = test_model(model, dataloaders, device, config["input_size"])

    # printing the results on test dataset
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == "__main__":
    run_script()