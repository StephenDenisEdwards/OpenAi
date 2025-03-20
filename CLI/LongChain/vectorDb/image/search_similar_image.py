import faiss
import numpy as np
import torch
import clip
from PIL import Image
import os
import pickle  # New import

# OpenMP Conflict (libomp140.x86_64.dll vs libiomp5md.dll)
# This happens because multiple OpenMP versions are being loaded, often due to conflicting installs of FAISS, PyTorch, and NumPy.

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Step 1: Load Images and Extract Embeddings
# We use OpenAI’s CLIP model to generate 512-dimensional embeddings for images.

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Directory containing images
image_folder = "images_dataset"
image_files = []
for root, dirs, files in os.walk(image_folder):
    for f in files:
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            image_files.append(os.path.join(root, f))

# Persistence file paths
persist_index_path = "search_similarity_index.faiss"
persist_info_path = "search_similarity_data.pkl"

# Function to compute image embeddings
def get_image_embedding(image_path):
    print("Processing: ", image_path)
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)  # Preprocess and batchify
    with torch.no_grad():
        embedding = model.encode_image(image)
    return embedding.cpu().numpy()

if os.path.exists(persist_index_path) and os.path.exists(persist_info_path):
    # Persisted data found. Load the FAISS index and image map.
    print("Persisted data found. Loading index and image information...")
    index = faiss.read_index(persist_index_path)
    with open(persist_info_path, "rb") as f:
        image_index_map = pickle.load(f)
else:
    # No persisted data. Process images and perform training.
    # Compute embeddings for all images
    image_embeddings = np.array([get_image_embedding(img) for img in image_files])
    image_embeddings = image_embeddings.reshape(len(image_files), -1).astype(np.float32)  # Flatten
    # Normalize embeddings for cosine similarity
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    
    # Store image file names with indices
    image_index_map = {i: image_files[i] for i in range(len(image_files))}
    
    # FAISS settings
    d = image_embeddings.shape[1]  # Dimensionality (512 for CLIP)
    nlist = 5  # Number of clusters
    nprobe = 10  # Increased number of clusters to search for better matching
    
    # Create FAISS index with inner product (for cosine similarity)
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # Train the FAISS index (needed for IVFFlat)
    index.train(image_embeddings)
    index.add(image_embeddings)  # Add vectors to FAISS
    index.nprobe = nprobe  # Set nprobe for improved recall
    
    # Persist trained data to disk
    faiss.write_index(index, persist_index_path)
    with open(persist_info_path, "wb") as f:
        pickle.dump(image_index_map, f)

# Step 3: Search for Similar Images
# To find similar images, extract an embedding for a query image and search in FAISS.

def search_similar_images(query_image_path, k=3):
    query_embedding = get_image_embedding(query_image_path).reshape(1, -1).astype(np.float32)
    # Normalize query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    # Set nprobe for improved search (in case index was loaded)
    index.nprobe = 10
    # Perform search in FAISS
    distances, indices = index.search(query_embedding, k)
    
    print(f"\nQuery Image: {query_image_path}\n")
    for i, idx in enumerate(indices[0]):
        print(f"    Match {i+1}: {image_index_map[idx]} (Score: {distances[0][i]:.4f})")


# Example: Search for similar images
search_similar_images("query\query_dog.jpg", 10)
search_similar_images("query\query_car.jpg", 10)
search_similar_images("query\query_dog.jpg", 10)
search_similar_images("query\query_airplane.jpg", 10)
search_similar_images("query\childlike_airplane_drawing_6yrs.jpg", 10)
search_similar_images("query\childlike_airplane_drawing_12yrs.jpg", 10)
