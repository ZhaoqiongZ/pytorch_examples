import torch
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast
from PIL import Image
import json

# Load a pre-trained ResNet18 model
model = torchvision.models.resnet18(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Move the model to Intel GPU and convert to FP16
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
model.to(device)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load an image
image = Image.open("test_image_classification.jpg")
image = transform(image).unsqueeze(0)  # Add batch dimension
image = image.to(device)  # Move image to Intel GPU

# Perform inference with autocast
with torch.no_grad():
    with torch.autocast(device_type="xpu", dtype=torch.float16, enabled=True):
        output = model(image)

# Get the predicted class
_, predicted = torch.max(output, 1)
print("Predicted class index:", predicted.item())

# Load ImageNet class index
with open('imagenet_class_index.json') as f:
    class_idx = json.load(f)

# Get the class name
class_name = class_idx[str(predicted.item())][1]
print("Predicted class name:", class_name)

# Get top 5 predictions
topk = 5
_, indices = torch.topk(output, topk)
top5_classes = [class_idx[str(idx.item())][1] for idx in indices[0]]
print("Top 5 predictions:", top5_classes)