from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # Load the data using ImageFolder
    data_transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_dataset = torchvision.datasets.ImageFolder(root='../images',
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

    # transfer the dataset into the sklearn format
    train_data = []
    train_labels = []
    for data in train_dataset:
        inputs, labels = data
        images = inputs.numpy().reshape(-1)
        train_data.append(images)
        train_labels.append([labels])

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    test_data = []
    test_labels = []

    for i, data in enumerate(val_dataset, 0):
        inputs, labels = data
        images = inputs.numpy().reshape(-1)
        test_data.append(images)
        test_labels.append(labels)

    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    # train the model
    models = [RandomForestClassifier(random_state=45), DecisionTreeClassifier(random_state=0)]
    results = []
    for m in models:
        m.fit(train_data, train_labels)
        predictions = m.predict(test_data)
        accuracy = accuracy_score(test_labels, predictions)
        print("Models: ", type(m), "Accuracy: ", accuracy)
        results.append(accuracy)
    best_model = models[np.argmax(results)]
    print("The best model is", type(best_model))
    # save the model
    path = '../model/sklearn_best_model.pkl'
    joblib.dump(best_model, path)
    # load the model from disk
    loaded_model = joblib.load(path)
    result = loaded_model.score(test_data, test_labels)
    print("Load Accuracy: ", result)
