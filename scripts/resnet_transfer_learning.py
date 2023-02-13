# library imports
import torch
import torch.nn as nn
import torch.optim as optim
import json
from PIL import Image
Image.MAX_IMAGE_PIXELS = 120000000

# local imports
from resnet_helper_functions import create_datasets, create_dataloaders, train_model

# torch parameters being used
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

def run_script():
    """
    Driving function for the script. Trains a model on the training set while evaluating it on the validation set.
    Saves the best model 

    """

    # read in config file
    with open('config.json') as f:
        config = json.load(f)

    # instantiate pre-trained resnet
    net = torch.hub.load('pytorch/vision', config["pretrained_model_to_use"], weights='imagenet')
    
    # shut off autograd for all layers to freeze model so the layer weights are not trained
    if config["freeze_pretrained_model"]:
        for param in net.parameters():
            param.requires_grad = False

    # get the number of inputs to final FC layer
    num_ftrs = net.fc.in_features

    # replace existing FC layer with a new FC layer having the same number of inputs and num_classes outputs
    net.fc = nn.Linear(num_ftrs, num_classes)

    # cross entropy loss combines softmax and nn.NLLLoss() in one single class.
    criterion = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = optim.Adam(net.fc.parameters(), lr=0.001)

    # learning rate scheduler - not using as we used adam optimizer
    lr_scheduler = None

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create datasets and dataloaders
    train_dataset, val_dataset, test_dataset = create_datasets(config["data_dir"], config["train_perc"], config["val_perc"], config["test_perc"])
    dataloaders, dataset_sizes, class_names, num_classes = create_dataloaders(train_dataset, val_dataset, batch_size=config["batch_size"])

    # train the model
    model_dir = config["model_dir"]
    num_epochs = config["num_epochs"]
    net = train_model(model = net, model_dir = model_dir, criterion = criterion, optimizer = optimizer, dataloaders = dataloaders, dataset_sizes = dataset_sizes, lr_scheduler = lr_scheduler, device = device, num_epochs = num_epochs)

    # test the model
    #test_model(model = net, test_dataset = test_dataset, device = device)

if __name__ == '__main__':
    run_script()