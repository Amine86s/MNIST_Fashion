import sys
sys.path.append('.')
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import FashionCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# DATA
train_set = torchvision.datasets.FashionMNIST("./data", download=True,
                       transform=transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST("./data", train=False, download=True,
                       transform=transforms.Compose([transforms.ToTensor()]))

train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

# MODEL
model = FashionCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
count = 0

loss_list = []
iteration_list = []
accuracy_list = []

for epoch in range(num_epochs):
    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)
        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)

        outputs = model(train)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count += 1

        # TEST EVERY 50 STEPS
        if not (count % 50):
            total = 0
            correct = 0

            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                test = Variable(images.view(100, 1, 28, 28))
                outputs = model(test)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

                total += labels.size(0)

            accuracy = 100 * correct / total

            loss_list.append(loss.item())
            iteration_list.append(count)
            accuracy_list.append(accuracy)

            print(f"Iteration: {count} Loss: {loss.item():.4f} Accuracy: {accuracy:.2f}%")

# SAVE MODEL
torch.save(model.state_dict(), "models/fashion_cnn.pth")
print("\nModel saved to: models/fashion_cnn.pth")

# PLOTS
plt.plot(iteration_list, loss_list)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.savefig("results/loss.png")
plt.close()

plt.plot(iteration_list, accuracy_list)
plt.xlabel("Iterations")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Curve")
plt.savefig("results/accuracy.png")
plt.close()

print("\nTraining complete.")
