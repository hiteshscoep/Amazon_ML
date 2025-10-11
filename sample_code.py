import pandas as pd
import re
from sentence_transformers import SentenceTransformer

# --- Step 1: Load training data ---
DATASET_FOLDER = 'dataset/'
train_df = pd.read_csv(DATASET_FOLDER + 'train.csv')

# --- Step 2: Optional text cleaning function ---
def clean_text(text):
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Apply cleaning to catalog content
train_texts = train_df['catalog_content'].apply(clean_text).tolist()

# --- Step 3: Load pretrained text embedding model ---
# Lightweight, 384-dimensional embeddings, fast for large datasets
text_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')  # use 'cuda' if GPU available

# --- Step 4: Compute embeddings for all training texts ---
print("Computing text embeddings for training set...")
text_embeddings = text_model.encode(
    train_texts,
    batch_size=64,           # adjust based on your RAM
    show_progress_bar=True
)
print("Done computing embeddings.")

# --- Step 5: Check shape ---
print("Shape of text embeddings:", text_embeddings[0])
# Expected output: (75000, 384)
