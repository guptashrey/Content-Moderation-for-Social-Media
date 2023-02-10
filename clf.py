from torchvision import transforms
import torch
from PIL import Image
import urllib.request
import os

if not os.path.isfile('./model/best_model.pt'):
    urllib.request.urlretrieve("https://duke.box.com/shared/static/yrzky3rmcv7fj3txeer375br6nuvm5ly.pt", "./model/best_model.pt")

model = torch.load('./model/best_model.pt', map_location=torch.device('cpu'))
model.eval()

with open('./model/classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

def predict(image_path):

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)
    out = model(batch_t)

    prob = torch.nn.functional.softmax(out, dim=1)[0]
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0]]