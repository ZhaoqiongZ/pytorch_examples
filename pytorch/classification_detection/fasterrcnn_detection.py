import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.amp import autocast
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Move the model to Intel GPU
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
model.to(device)

# Define image preprocessing steps
def preprocess(image):
    image = F.to_tensor(image)
    return image

# Load and preprocess the image
image_path = "test_image_detection.jpg"
image = Image.open(image_path).convert("RGB")
image = preprocess(image).unsqueeze(0)  # Add batch dimension
image = image.to(device)

# Perform inference with autocast
with torch.no_grad():
    with torch.autocast(device_type="xpu", dtype=torch.float16, enabled=True):
        outputs = model(image)

# Get prediction results
outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

# Visualize results
fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(image.squeeze(0).permute(1, 2, 0).cpu().numpy())

# Draw bounding boxes with different colors for each box
colors = []
for i in range(len(outputs[0]['boxes'])):
    colors.append((random.random(), random.random(), random.random()))

for box, score, label, color in zip(outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels'], colors):
    if score > 0.7:  # Only show detections with confidence greater than 0.7
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min, f'{label.item()}: {score:.2f}', bbox=dict(facecolor=color, alpha=0.5))

# Save and show the result image
output_image_path = "output_image.jpg"
plt.savefig(output_image_path)
plt.show()
plt.close()

print(f"Detection results saved to {output_image_path}")