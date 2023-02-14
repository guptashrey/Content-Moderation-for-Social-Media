# library imports
import torch
import torchvision
import torchvision.transforms as transforms
import time
import copy

def define_transforms():
    """
    Define transformations for training, validation, and test datasets.
    training data: resize to 256 * 256, center cropping, randomized horizontal & vertical flipping, and normalization
    validation and test data: resize to 256 * 256, center cropping, and normalization
    """
#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'test': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#     }

    data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    return data_transforms

def create_datasets(data_dir, train_perc, val_perc, test_perc):
    """
    Create datasets for training, validation, and test

    Args:
        data_dir (str): path to data directory

    Returns:
        train_dataset (torchvision.datasets.ImageFolder): training dataset
        val_dataset (torchvision.datasets.ImageFolder): validation dataset
        test_dataset (torchvision.datasets.ImageFolder): test dataset
        class_names (list): list of class names
        num_classes (int): number of classes
    """
    ## Define transformations for training, validation, and test data
    data_transforms = define_transforms()

    ## Create Datasets for training and validation sets
    image_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=data_transforms)
    train_size = int(train_perc * len(image_dataset))
    val_size = int(val_perc * len(image_dataset))
    test_size = len(image_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(image_dataset, [train_size, val_size, test_size])
    
    # get class names associated with labels
    class_names = image_dataset.classes
    num_classes = len(class_names)

    return train_dataset, val_dataset, test_dataset, class_names, num_classes


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers=2):
    """
    Create dataloaders for training, validation and test datasets

    Args:
        train_dataset (torchvision.datasets.ImageFolder): training dataset
        val_dataset (torchvision.datasets.ImageFolder): validation dataset
        test_dataset (torchvision.datasets.ImageFolder): test dataset
        batch_size (int): batch size

    Returns:
        dataloaders (dict): dictionary of dataloaders for training and validation sets
        dataset_sizes (dict): dictionary of sizes of training and validation sets
    """
    # create DataLoaders for training and validation sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # set up dict for dataloaders
    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    # store size of training and validation sets
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

    return dataloaders, dataset_sizes

def train_model(model, model_dir, criterion, optimizer, dataloaders, dataset_sizes, scheduler=None, device="cpu", num_epochs=1):
    """
    Train the model using transfer learning

    Args:
        model (torchvision.models): model to train
        model_dir (str): path to directory to save model
        criterion (torch.nn.modules.loss): loss function
        optimizer (torch.optim): optimizer
        dataloaders (dict): dictionary of dataloaders for training and validation sets
        dataset_sizes (dict): dictionary of sizes of training and validation sets
        scheduler (torch.optim.lr_scheduler): learning rate scheduler
        device (torch.device): device to train on
        num_epochs (int): number of epochs to train for

    Returns:
        model (torchvision.models): trained model
    """
    # load the model to GPU if available
    model = model.to(device) 
    since = time.time()

    # initialize best model weights and accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # loop over epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval()   # set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # iterate the data loader
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the weight gradients
                optimizer.zero_grad()

                # forward pass to get outputs and calculate loss and track gradient only for training data
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backpropagation to get the gradients with respect to each weight only if in train phase
                    if phase == 'train':
                        loss.backward()
                        # update the weights
                        optimizer.step()

                # convert loss into a scalar and add it to running_loss
                running_loss += loss.item() * inputs.size(0)
                # track number of correct predictions
                running_corrects += torch.sum(preds == labels.data)

            # step along learning rate scheduler when in train
            if phase == 'train':
                if scheduler != None:
                    scheduler.step()

            # calculate and display average loss and accuracy for the epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(phase.capitalize(), epoch_loss, epoch_acc))

            # if model performs better on val set, save weights as the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, model_dir)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Accuracy: {:3f}'.format(best_acc))

    # load the weights from best model
    model.load_state_dict(best_model_wts)
    return model

def test_model(model, dataloaders, device):
    """
    Test the trained model performance on test dataset

    Args:
        model (torchvision.models): model to train
        dataloaders (dict): dictionary of dataloaders for training, validation and test sets

    Returns:
        model (torchvision.models): trained model
    """
    model = model.to(device)

    # set model to evaluate mode
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    return all_preds, all_labels