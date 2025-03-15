import faiss
import numpy as np
import torch
import clip
from PIL import Image
import os

# Step 1: Load Images and Extract Embeddings
# We use OpenAI’s CLIP model to generate 512-dimensional embeddings for images.

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Directory containing images
image_folder = "images_dataset"
image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png", ".jpeg"))]

# Function to compute image embeddings
def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)  # Preprocess and batchify
    with torch.no_grad():
        embedding = model.encode_image(image)
    return embedding.cpu().numpy()

# Compute embeddings for all images
image_embeddings = np.array([get_image_embedding(os.path.join(image_folder, img)) for img in image_files])
image_embeddings = image_embeddings.reshape(len(image_files), -1).astype(np.float32)  # Flatten

# Store image file names with indices
image_index_map = {i: image_files[i] for i in range(len(image_files))}


# Step 2: Indexing in FAISS (Approximate Nearest Neighbor)
# We use FAISS’s IndexIVFFlat, which speeds up search by clustering vectors.

# FAISS settings
d = image_embeddings.shape[1]  # Dimensionality (512 for CLIP)
nlist = 5  # Number of clusters (adjustable for performance)
nprobe = 2  # Number of clusters to search (higher = more accurate)

# Create FAISS index with clustering (ANN)
quantizer = faiss.IndexFlatL2(d)  # Base index for clustering
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

# Train the FAISS index (needed for IVFFlat)
index.train(image_embeddings)
index.add(image_embeddings)  # Add vectors to FAISS

# Step 3: Search for Similar Images
# To find similar images, extract an embedding for a query image and search in FAISS.

def search_similar_images(query_image_path, k=3):
    query_embedding = get_image_embedding(query_image_path).reshape(1, -1).astype(np.float32)
    
    # Perform search in FAISS
    distances, indices = index.search(query_embedding, k)
    
    print(f"\nQuery Image: {query_image_path}\n")
    for i, idx in enumerate(indices[0]):
        print(f"Match {i+1}: {image_index_map[idx]} (Distance: {distances[0][i]:.4f})")

# Example: Search for similar images
query_image = "query.jpg"  # Replace with your query image
search_similar_images(query_image)
