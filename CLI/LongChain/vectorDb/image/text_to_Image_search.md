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


# Great question! The **Text-to-Image Search Using FAISS and CLIP** works **without labels** because **CLIP (Contrastive Language-Image Pretraining)** learns a **shared embedding space** for both images and text.

### **How Does CLIP Enable Text Search Without Labels?**

1. **Pretrained Vision-Language Model** :

* **CLIP** has been trained on  **millions of image-text pairs** .
* It learns  **associations between images and textual descriptions** .
* The model understands **semantic meaning** without explicit labels.

1. **Image Embeddings (Vector Representation of Images)** :

* Each image is converted into a **512-dimensional vector** (using `model.encode_image`).
* These embeddings capture  **features like colors, objects, and scene types** .
* The images are  **indexed in FAISS** , which enables  **fast similarity search** .

1. **Text Query as a Vector** :

* When you input a **text query** (e.g.,  *"a red sports car"* ), CLIP converts it into a  **512-dimensional vector** .
* The model **understands the query’s meaning** even though images were not pre-labeled.

1. **Similarity Search** in FAISS:
   * FAISS **compares the text query embedding** to the indexed image embeddings.
   * It **retrieves the most similar images** based on  **vector distance (cosine similarity or L2 norm)** .

---

### **Why Does This Work Without Labels?**

* **CLIP does not need explicit image labels** like "dog" or "car".
* Instead, it **generalizes meaning** based on learned image-text pairs.
* It recognizes **visual features** that match textual descriptions.

---

### **Example Breakdown**

#### **Step 1: Encoding and Indexing Images**

```python
# Encode images (512D vector)
image_embeddings = model.encode_image(images)
index.add(image_embeddings)  # Store in FAISS
```

* Each image is stored **without labels** but represented as a  **vector** .

#### **Step 2: Encoding and Searching a Text Query**

```python
query_text = "a red sports car"
query_embedding = model.encode_text(clip.tokenize([query_text]).to(device))

# Search for similar images
distances, indices = index.search(query_embedding.cpu().numpy(), k=3)
```

* **CLIP understands the text** and converts it into an  **embedding** .
* FAISS finds the  **closest image vectors** .

---

### **Summary**

* No need for  **manual labels** —CLIP already understands  **semantic meaning** .
* Text queries work by  **comparing vector embeddings** , not pre-defined labels.
* FAISS efficiently retrieves  **similar images based on learned features** .

Would you like a **visual example** or **step-by-step debugging guide** to better understand how embeddings are matched? 🚀
