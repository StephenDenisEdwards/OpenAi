import faiss
import numpy as np
import torch
import clip
from PIL import Image
import os
import pickle

# OpenMP Conflict (libomp140.x86_64.dll vs libiomp5md.dll)
# This happens because multiple OpenMP versions are being loaded, often due to conflicting installs of FAISS, PyTorch, and NumPy.

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Step 1: Install Required Libraries

# Step 2: Load Images and Compute Embeddings
# We use CLIP (ViT-B/32) to encode images into 512-dimensional vectors.

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Directory containing images
image_folder = "images_dataset"
image_files = []

# Define persistence file paths
index_filepath = "text_to_image_index.index"
mapping_filepath = "text_to_image_mapping.pkl"

if os.path.exists(index_filepath) and os.path.exists(mapping_filepath):
    print("Loading persisted index and mapping.")
    index = faiss.read_index(index_filepath)
    with open(mapping_filepath, "rb") as f:
        image_index_map = pickle.load(f)
else:
    for root, dirs, files in os.walk(image_folder):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                image_files.append(os.path.join(root, f))

    # Function to compute image embeddings
    def get_image_embedding(image_path):
        print("Processing: ", image_path)
        image = Image.open(image_path).convert("RGB")
        image = preprocess(image).unsqueeze(0).to(device)  # Preprocess and batchify
        with torch.no_grad():
            embedding = model.encode_image(image)
        return embedding.cpu().numpy()

    # Compute embeddings for all images
    image_embeddings = np.array([get_image_embedding(img) for img in image_files])
    image_embeddings = image_embeddings.reshape(len(image_files), -1).astype(np.float32)  # Flatten

    # Store image file names with indices
    image_index_map = {i: image_files[i] for i in range(len(image_files))}

    # Step 3: Index Image Embeddings in FAISS
    # We use FAISS with IndexFlatL2 (Euclidean distance) for fast retrieval.
    # FAISS settings
    d = image_embeddings.shape[1]  # 512-dimensional embeddings

    # Create FAISS index (FlatL2 = exact search)
    index = faiss.IndexFlatL2(d)
    index.add(image_embeddings)  # Add vectors to FAISS

    # Persist the index and mapping
    faiss.write_index(index, index_filepath)
    with open(mapping_filepath, "wb") as f:
        pickle.dump(image_index_map, f)


def search_by_text(query_text, k=3):
    # Encode text query into an embedding
    with torch.no_grad():
        text_embedding = model.encode_text(clip.tokenize([query_text]).to(device)).cpu().numpy()
    
    text_embedding = text_embedding.astype(np.float32)
    
    # Perform search in FAISS
    distances, indices = index.search(text_embedding, k)
    
    print(f"\nQuery: \"{query_text}\"\n")
    for i, idx in enumerate(indices[0]):
        print(f"Match {i+1}: {image_index_map[idx]} (Distance: {distances[0][i]:.4f})")

# Example: Search with a text prompt
# search_by_text("a red sports car")

search_by_text("a cute dog")

