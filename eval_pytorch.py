import torch
import torch.nn.functional as F
from struct import unpack
import gzip
import numpy as np

# ==== 1. Load Test Data ====
test_images_path = './dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = './dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

# Normalize and reshape
test_imgs = test_imgs / 255.0
test_imgs = test_imgs.reshape(-1, 1, 28, 28).astype(np.float32)

# Convert to PyTorch tensor
test_imgs_tensor = torch.tensor(test_imgs, dtype=torch.float32)
test_labs_tensor = torch.tensor(test_labs, dtype=torch.long)

# ==== 2. Load the trained model ====
from train_pytorch import CNN 

model = CNN()
model.load_state_dict(torch.load('./best_models/best_model_pytorch.pth'))
model.eval()  # very important!

# ==== 3. Forward and Evaluation ====
with torch.no_grad():
    outputs = model(test_imgs_tensor)  # shape: (batch_size, num_classes)
    preds = outputs.argmax(dim=1)    
    acc = (preds == test_labs_tensor).float().mean().item()

print(f"Test Accuracy: {acc*100:.2f}%")
