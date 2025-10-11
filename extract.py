import os
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
from pathlib import Path
from functools import partial
import requests

# --- For text embeddings ---
from sentence_transformers import SentenceTransformer

# --- For image embeddings ---
import torch
from torchvision import transforms
from PIL import Image
import timm

# ----------------------------
# 1️⃣ Load training data
# ----------------------------
DATASET_FOLDER = 'dataset/'
train_csv_path = os.path.join(DATASET_FOLDER, 'train.csv')
train_df = pd.read_csv(train_csv_path)
print("Train dataset loaded:", train_df.shape)

# ----------------------------
# 2️⃣ Download images (requests with verify=False)
# ----------------------------
def download_image(image_link, savefolder):
    if isinstance(image_link, str):
        filename = Path(image_link).name
        image_save_path = os.path.join(savefolder, filename)
        if not os.path.exists(image_save_path):
            try:
                r = requests.get(image_link, timeout=10, verify=False)  # ignore SSL
                with open(image_save_path, 'wb') as f:
                    f.write(r.content)
            except Exception as ex:
                print(f'Warning: Not able to download - {image_link}\n{ex}')
    return

def download_images(image_links, download_folder):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    download_image_partial = partial(download_image, savefolder=download_folder)
    for link in tqdm(image_links, desc="Downloading images"):
        download_image_partial(link)

# Extract image URLs from train.csv
image_links = train_df['image_link'].tolist()
image_folder = os.path.join(DATASET_FOLDER, 'images/')
download_images(image_links, image_folder)
print("All images downloaded to:", image_folder)

# ----------------------------
# 3️⃣ Text Embeddings
# ----------------------------
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

train_texts = train_df['catalog_content'].apply(clean_text).tolist()

text_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')  # 'cuda' if available

print("Computing text embeddings...")
text_embeddings = text_model.encode(
    train_texts,
    batch_size=64,
    show_progress_bar=True
)
print("Text embeddings shape:", text_embeddings.shape)

# Save text embeddings
np.save(os.path.join(DATASET_FOLDER, 'text_embeddings.npy'), text_embeddings)
print("Text embeddings saved to text_embeddings.npy")

# ----------------------------
# 4️⃣ Image Embeddings
# ----------------------------
# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        return preprocess(img)
    except:
        return None

# Load pretrained model (ResNet50)
image_model = timm.create_model('resnet50', pretrained=True)
image_model = torch.nn.Sequential(*list(image_model.children())[:-1])  # remove classifier
image_model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_model.to(device)

# Compute image embeddings
image_embeddings = []
print("Computing image embeddings...")
for img_file in tqdm(os.listdir(image_folder), desc="Processing images"):
    img_path = os.path.join(image_folder, img_file)
    img_tensor = load_image(img_path)
    if img_tensor is not None:
        img_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = image_model(img_tensor)
        emb = emb.squeeze().cpu().numpy()
        image_embeddings.append(emb)
image_embeddings = np.array(image_embeddings)
print("Image embeddings shape:", image_embeddings.shape)

# Save image embeddings
np.save(os.path.join(DATASET_FOLDER, 'image_embeddings.npy'), image_embeddings)
print("Image embeddings saved to image_embeddings.npy")
