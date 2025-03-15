### **Text-to-Image Search Using FAISS and CLIP**

We will:

1. **Extract image embeddings** using OpenAI's **CLIP** model.
2. **Index these embeddings in FAISS** for fast retrieval.
3. **Use a text query** (e.g.,  *"a red sports car"* ) to find the most relevant images.

---

### **📌 Step 1: Install Required Libraries**

If you haven't installed them yet, run:

```bash
pip install faiss-cpu torch torchvision numpy pillow openai-clip
```

---

### **📌 Step 2: Load Images and Compute Embeddings**

We use **CLIP (ViT-B/32)** to encode images into  **512-dimensional vectors** .

```python
import faiss
import numpy as np
import torch
import clip
from PIL import Image
import os

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
```

---

### **📌 Step 3: Index Image Embeddings in FAISS**

We use **FAISS with `IndexFlatL2`** (Euclidean distance) for fast retrieval.

```python
# FAISS settings
d = image_embeddings.shape[1]  # 512-dimensional embeddings

# Create FAISS index (FlatL2 = exact search)
index = faiss.IndexFlatL2(d)
index.add(image_embeddings)  # Add vectors to FAISS
```

---

### **📌 Step 4: Search for Images Using Text Queries**

We encode a **text prompt** into the same vector space as the images.

```python
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
search_by_text("a red sports car")
```

---

### **📌 Expected Output**

If the dataset contains images of  **cars, animals, landscapes** , and the user searches for `"a red sports car"`, FAISS will return the most visually similar images:

```
Query: "a red sports car"

Match 1: ferrari.jpg (Distance: 0.1247)
Match 2: lamborghini.png (Distance: 0.1303)
Match 3: porsche.jpeg (Distance: 0.1351)
```

---

### **💡 Why This Works**

* **CLIP aligns text and image embeddings in the same space** , allowing text-to-image retrieval.
* **FAISS efficiently finds nearest neighbors** in high-dimensional spaces.
* **Exact search (`IndexFlatL2`) is good for small datasets** , but can be replaced with  **ANN methods (like HNSW or IVFFlat) for speed** .

---

### **🚀 Enhancements**

* **Use Approximate Nearest Neighbors (ANN)** : Replace `IndexFlatL2` with `IndexIVFFlat` for faster retrieval.
* **Use CLIP with Larger Models** : Try `ViT-L/14` for better accuracy.
* **Hybrid Search** : Combine FAISS with  **metadata filters (like Elasticsearch)** .

Would you like to see **an ANN-optimized version** of this for  **millions of images** ? 🚀
