import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.cuda import get_device_name
from torch import tensor
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':


    # Load the data using ImageFolder
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_dataset = torchvision.datasets.ImageFolder(root='./images',
                                                     transform=data_transform)

    # Split the data into training and validation sets
    train_size = int(0.9 * len(image_dataset))
    val_size = len(image_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(image_dataset, [train_size, val_size])

    # Create data loaders for the training and validation sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                               shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32,
                                             shuffle=True, num_workers=4)

    # Set up dict for dataloaders
    dataloaders = {'train': train_loader, 'val': val_loader}

    # Store size of training and validation sets
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    print(dataset_sizes)
    # Get class names associated with labels
    class_names = image_dataset.classes

    # Load the ResNet18 model
    model = torchvision.models.resnet18(pretrained=True)

    # Freeze the model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully-connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = "cuda:0"
    if torch.backends.mps.is_available():
        device_name = "mps"

    # Move the model to the GPU
    device = torch.device(device_name)
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Decay the learning rate by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Keep track of the best model's performance on the validation set
    best_acc = 0.0
    best_model_wts = model.state_dict()


    def train_model(model, criterion, optimizer, dataloaders, scheduler, device, num_epochs=25):
        model = model.to(device)  # Send model to GPU if available
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Get the input images and labels, and send to GPU if available
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Zero the weight gradients
                    optimizer.zero_grad()

                    # Forward pass to get outputs and calculate loss
                    # Track gradient only for training data
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backpropagation to get the gradients with respect to each weight
                        # Only if in train
                        if phase == 'train':
                            loss.backward()
                            # Update the weights
                            optimizer.step()

                    # Convert loss into a scalar and add it to running_loss
                    running_loss += loss.item() * inputs.size(0)
                    # Track number of correct predictions
                    running_corrects += torch.sum(preds == labels.data)

                # Step along learning rate scheduler when in train
                if phase == 'train':
                    scheduler.step()

                # Calculate and display average loss and accuracy for the epoch
                epoch_loss = running_loss / dataset_sizes[phase]
                # print(get_device_name(device))
                if device_name == 'mps':
                    running_corrects = tensor(running_corrects, device='mps', dtype=torch.float32)
                epoch_acc = running_corrects / dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # If model performs better on val set, save weights as the best model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, './model/best_model.pt')

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:3f}'.format(best_acc))

        # Load the weights from best model
        model.load_state_dict(best_model_wts)

        return model


    net = train_model(model, criterion, optimizer, dataloaders, scheduler, device, num_epochs=10)


    # Load the saved state_dict into the model
    # model.load_state_dict(torch.load('./model/best_model.pt'))
    # Set the model to evaluation mode
    # model.eval()

    # Display a batch of predictions
    def visualize_results(model, dataloader, device):
        model = model.to(device)  # Send model to GPU if available
        with torch.no_grad():
            model.eval()
            # Get a batch of validation images
            images, labels = next(iter(dataloader))
            images, labels = images.to(device), labels.to(device)
            # Get predictions
            _, preds = torch.max(model(images), 1)
            preds = np.squeeze(preds.cpu().numpy())
            images = images.cpu().numpy()

        # Plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(40, 10))
        for idx in np.arange(len(preds) // 4):
            ax = fig.add_subplot(2, len(preds) // 2, idx + 1, xticks=[], yticks=[])
            image = images[idx]
            image = image.transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)
            ax.imshow(image)
            ax.set_title("{} ({})".format(class_names[preds[idx]], class_names[labels[idx]]),
                         color=("green" if preds[idx] == labels[idx] else "red"))
        return


    visualize_results(net, val_loader, device)
