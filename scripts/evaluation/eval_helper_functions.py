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

def test_svm_model(model, dataloaders, device, input_size):
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

            # Reshape images
            inputs = inputs.reshape(-1, input_size)
            # Forward pass
            outputs = model(inputs)
            # Get predictions
            preds = torch.argmax(outputs, axis=1)

            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    return all_preds, all_labels