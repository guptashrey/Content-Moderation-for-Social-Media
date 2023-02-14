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
from resnet_helper_functions import create_datasets, create_dataloaders, train_model, test_model

# torch parameters being used
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
logging.info(f'Torch Version: {TORCH_VERSION}, Cuda Version: {CUDA_VERSION}')

def run_script():
    """
    Driving function for the script. Trains a model on the training set while evaluating it on the validation set.
    Saves the best model 

    """

    # read in config file
    with open('resnet_config.json') as f:
        config = json.load(f)
        logging.info('Resnet Config File Read Successfully.')
        
    # create datasets and dataloaders
    train_dataset, val_dataset, test_dataset, class_names, num_classes = create_datasets(config["data_dir"], config["train_perc"], config["val_perc"], config["test_perc"])
    logging.info('Train, Validation and Test Datasets Created Successfully.')
    
    dataloaders, dataset_sizes = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=config["batch_size"], num_workers = config["num_workers"])
    logging.info('Train, Validation and Test Dataloaders Created Successfully.')

    # instantiate pre-trained resnet
    #net = torch.hub.load('pytorch/vision', config["pretrained_model_to_use"], weights=config["weights_to_use"])
    if config["pretrained_model_to_use"] == 'resnet18':
        net = torchvision.models.resnet18(pretrained=True)
    elif config["pretrained_model_to_use"] == 'resnet34':
        net = torchvision.models.resnet34(pretrained=True)
    elif config["pretrained_model_to_use"] == 'resnet50':
        net = torchvision.models.resnet50(pretrained=True)
    elif config["pretrained_model_to_use"] == 'resnet101':
        net = torchvision.models.resnet101(pretrained=True)
    elif config["pretrained_model_to_use"] == 'resnet152':
        net = torchvision.models.resnet152(pretrained=True)
    
    logging.info('Model Loaded Successfully.')
    
    # shut off autograd for all layers to freeze model so the layer weights are not trained
    if config["freeze_pretrained_model"]:
        for param in net.parameters():
            param.requires_grad = False
    
    # get the number of inputs to final FC layer
    num_ftrs = net.fc.in_features

    # replace existing FC layer with a new FC layer having the same number of inputs and num_classes outputs
    net.fc = nn.Linear(num_ftrs, num_classes)
    
    # show model architecture
    temp, temp_ = next(iter(dataloaders['train']))
    logging.info('\nModel Architecture:')
    print(summary(net,(temp.shape[1:]), batch_size=config["batch_size"], device="cpu"), "\n")

    # cross entropy loss combines softmax and nn.NLLLoss() in one single class.
    criterion = nn.CrossEntropyLoss()

    # define optimizer
    if config["freeze_pretrained_model"]:
        optimizer = optim.Adam(net.fc.parameters(), lr=0.001)
    else:
        optimizer = optim.Adam(net.parameters(), lr=0.001)

    # learning rate scheduler - not using as we used adam optimizer
    lr_scheduler = None

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Device {device} Being Used.')
    
    # train the model
    model_dir = config["model_dir"]
    num_epochs = config["num_epochs"]
    
    logging.info('Started Training The Model.\n')
    
    net = train_model(model = net, model_dir = model_dir, criterion = criterion, optimizer = optimizer, dataloaders = dataloaders, dataset_sizes = dataset_sizes, scheduler = lr_scheduler, device = device, num_epochs = num_epochs)

    # test the model
    all_preds, all_labels = test_model(net, dataloaders, device)

    # printing the results on test dataset
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == '__main__':
    run_script()