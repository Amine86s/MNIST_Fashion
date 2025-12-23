import torch
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, classification_report

import torchvision
import torchvision.transforms as transforms

from model import FashionCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_set = torchvision.datasets.FashionMNIST("./data", train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor()]))

test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)

model = FashionCNN().to(device)
model.load_state_dict(torch.load("models/fashion_cnn.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        test = Variable(images)
        outputs = model(test)

        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

print("\nConfusion Matrix:\n")
print(cm)

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds))

print("Evaluation complete.")
