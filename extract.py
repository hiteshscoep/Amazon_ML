# ==========================================
# Full pipeline: Text + Structured Features
# ==========================================

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv('dataset/test.csv')

# Fill missing fields
# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv('dataset/test.csv')

# -----------------------------
# 2. Handle missing columns
# -----------------------------
if 'catalog_content' not in df.columns:
    df['catalog_content'] = ''

if 'Value' not in df.columns:
    df['Value'] = 0.0

if 'Unit' not in df.columns:
    df['Unit'] = 'Ounce'  # default unit

# Fill missing values in existing columns
df['catalog_content'] = df['catalog_content'].fillna('')
df['Value'] = df['Value'].fillna(0.0)
df['Unit'] = df['Unit'].fillna('Ounce')


# -----------------------------
# 2. Helper functions
# -----------------------------

# Extract Item Name, Bullet Points, Product Description
def split_catalog_text(text):
    # Item Name
    name_match = re.search(r'Item Name:\s*(.*?)(?:Bullet Point|Product Description|$)', text, re.IGNORECASE|re.DOTALL)
    name = name_match.group(1).strip() if name_match else ''

    # Bullet Points (concatenated)
    bullets = re.findall(r'Bullet Point \d+: (.*?)(?=Bullet Point \d+:|Product Description|$)', text, re.IGNORECASE|re.DOTALL)
    bullets_text = ' '.join(bullets) if bullets else ''

    # Product Description
    desc_match = re.search(r'Product Description:\s*(.*?)(?=Value|Unit|$)', text, re.IGNORECASE|re.DOTALL)
    desc = desc_match.group(1).strip() if desc_match else ''

    # Combine all text fields into a single string for embedding
    combined_text = ' '.join([name, bullets_text, desc])
    return combined_text

# Extract weight from text (oz, fl oz, lb, pounds)
def extract_weight(text):
    match = re.search(r'(\d+(\.\d+)?)\s*(oz|fl oz|pounds|lb)', text, re.IGNORECASE)
    return float(match.group(1)) if match else 0.0

# Extract pack size from text (Pack of X, X Bars/pieces/Count)
def extract_pack_size(text):
    match = re.search(r'Pack of (\d+)', text, re.IGNORECASE)
    if match: 
        return int(match.group(1))
    match2 = re.findall(r'(\d+)\s*(Bars|pieces|Count)', text, re.IGNORECASE)
    return int(match2[0][0]) if match2 else 1

# Extract count based on unit type
def extract_count(unit, text):
    if unit.lower() == 'count':
        return extract_pack_size(text)
    return 1

# -----------------------------
# 3. Extract structured features
# -----------------------------
print("Extracting structured features...")

# Value (numeric)
value_feature = df['Value'].fillna(0.0).astype(float).values.reshape(-1,1)

# Unit (categorical → one-hot)
unit_feature = df['Unit'].values.reshape(-1,1)
unit_encoder = OneHotEncoder(sparse_output=False)
unit_onehot = unit_encoder.fit_transform(unit_feature)

# Weight, Pack size, Count
weight_feature = df['catalog_content'].apply(extract_weight).values.reshape(-1,1)
pack_feature = df['catalog_content'].apply(extract_pack_size).values.reshape(-1,1)
count_feature = np.array([extract_count(u,t) for u,t in zip(df['Unit'], df['catalog_content'])]).reshape(-1,1)

# Combine all structured features
structured_features = np.hstack([value_feature, weight_feature, pack_feature, count_feature, unit_onehot])
print("Structured features shape:", structured_features.shape)

# -----------------------------
# 4. Generate text embeddings
# -----------------------------
print("Generating text embeddings with all-mpnet-base-v2...")
model = SentenceTransformer('all-mpnet-base-v2')

# Combine all text fields safely
texts = df['catalog_content'].apply(split_catalog_text).tolist()

# Batch-wise embedding (memory efficient)
batch_size = 512
embeddings = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i:i+batch_size]
    batch_emb = model.encode(batch_texts, show_progress_bar=False)
    embeddings.append(batch_emb)

text_embeddings = np.vstack(embeddings)
print("Text embeddings shape:", text_embeddings.shape)

# -----------------------------
# 5. Fuse embeddings + structured
# -----------------------------
fused_features = np.hstack([text_embeddings, structured_features])
print("Fused features shape:", fused_features.shape)

# -----------------------------
# 6. Save to .npy
# -----------------------------
np.save('dataset/text_embeddings.npy', fused_features)
print("Saved fused features to fused_features.npy")
