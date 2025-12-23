import torch
from PIL import Image, ImageOps
import torchvision.transforms as transforms

from model import FashionCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = FashionCNN().to(device)
model.load_state_dict(torch.load("models/fashion_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


def predict(path):
    img = Image.open(path).convert("L")

    # INVERT IMAGE FOR MNIST COMPATIBILITY
    img = ImageOps.invert(img)

    tensor_img = transform(img)
    tensor_img = tensor_img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor_img)
        pred = torch.argmax(output, dim=1).cpu().item()

    print("Raw tensor:", output.argmax(1))
    print("Converted idx:", pred)

    return classes[pred]


