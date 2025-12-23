import torch
import gradio as gr
import torch.nn.functional as F
from PIL import Image, ImageOps
import torchvision.transforms as transforms

from model import FashionCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = [
    'T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
]

model = FashionCNN().to(device)
model.load_state_dict(torch.load("models/fashion_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

def predict(image):
    image = image.convert("L")
    image = ImageOps.invert(image)

    tensor_img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor_img)
        probs = F.softmax(output, dim=1)[0]

    return {classes[i]: float(probs[i]) for i in range(len(classes))}

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Giyim Ürünü Görüntüsü"),
    outputs=gr.Label(num_top_classes=3, label="Tahmin Olasılıkları"),
    title="🧠 Fashion-MNIST CNN Sınıflandırıcı",
    description=(
        "Bu uygulama, Fashion-MNIST veri seti ile eğitilmiş bir CNN modeli "
        "kullanarak yüklenen giyim ürününü sınıflandırır."
    )
)

if __name__ == "__main__":
    interface.launch(theme="soft")
