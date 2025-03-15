Here's an example of using **FAISS (Facebook AI Similarity Search)** to perform nearest neighbor searches on high-dimensional embeddings. We'll use `sentence-transformers` to generate embeddings from text and then index them with FAISS.

---

### **Steps in This Example:**

1. **Generate embeddings** using `sentence-transformers` (pre-trained model).
2. **Index the embeddings** in FAISS for fast similarity search.
3. **Perform a search** to find the most similar sentences.

---

### **Install Dependencies**

First, install the necessary libraries if you haven’t already:

```bash
pip install faiss-cpu sentence-transformers numpy
```

---

### **Code Implementation**

This script will:

* Encode sentences into high-dimensional vectors.
* Store them in a FAISS index.
* Retrieve the most similar sentences.

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load a pre-trained transformer model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample dataset (sentences to index)
sentences = [
    "Artificial intelligence is transforming the world.",
    "Machine learning is a subset of AI.",
    "FAISS is an efficient library for similarity search.",
    "Natural language processing is a fascinating field.",
    "Deep learning models require large amounts of data.",
    "Computers can now understand human speech."
]

# Convert sentences to embeddings (384-dimensional vectors)
embeddings = model.encode(sentences, normalize_embeddings=True)
embeddings = np.array(embeddings, dtype=np.float32)

# Create a FAISS index (L2 or cosine similarity search)
dimension = embeddings.shape[1]  # 384 for this model
index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
index.add(embeddings)  # Add vectors to the FAISS index

# Query sentence to search for similar ones
query_sentence = "AI is revolutionizing technology."
query_embedding = model.encode([query_sentence], normalize_embeddings=True)
query_embedding = np.array(query_embedding, dtype=np.float32)

# Search in FAISS
k = 3  # Number of nearest neighbors to retrieve
distances, indices = index.search(query_embedding, k)

# Print results
print(f"\nQuery: {query_sentence}\n")
for i, idx in enumerate(indices[0]):
    print(f"Rank {i+1}: {sentences[idx]} (Similarity: {distances[0][i]:.4f})")
```

---

### **How It Works**

1. We use **`sentence-transformers`** to generate 384-dimensional embeddings for the sentences.
2. We create a  **FAISS index (`IndexFlatIP`)** , which uses **inner product similarity** (equivalent to cosine similarity for normalized vectors).
3. A **query sentence** is embedded and searched in the FAISS index.
4. The **top `k` similar sentences** are retrieved along with their similarity scores.

---

### **Expected Output (Example)**

If you search with `"AI is revolutionizing technology."`, you might get:

```
Query: AI is revolutionizing technology.

Rank 1: Artificial intelligence is transforming the world. (Similarity: 0.8472)
Rank 2: Machine learning is a subset of AI. (Similarity: 0.8123)
Rank 3: Deep learning models require large amounts of data. (Similarity: 0.7785)
```

---

### **Extending This Approach**

* **For Large Datasets** : Use **FAISS with GPU acceleration** for massive-scale vector searches.
* **For Image Search** : Replace `sentence-transformers` with a  **pre-trained vision model (e.g., ResNet, CLIP)** .
* **For Hybrid Search** : Combine FAISS with  **metadata filters (e.g., Elasticsearch + FAISS)** .

Would you like an example of indexing **images** or using FAISS with  **approximate nearest neighbor (ANN) search for faster retrieval** ? 🚀


# `all-MiniLM-L6-v2`

The **`all-MiniLM-L6-v2`** model is a **pre-trained sentence transformer model** available on [Hugging Face’s Model Hub](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

### **How to Access the Model**

The model is automatically downloaded the first time you use it with the `sentence-transformers` library.

#### **1. Model Location After Download**

Once downloaded, it is stored in:

* **Linux/macOS:** `~/.cache/torch/sentence_transformers/`
* **Windows:** `C:\Users\YourUsername\.cache\torch\sentence_transformers\`

You can also manually specify a different cache directory using:

```python
from sentence_transformers import SentenceTransformer
import os

os.environ["TORCH_HOME"] = "/your/custom/path"
model = SentenceTransformer("all-MiniLM-L6-v2")
```

#### **2. Manually Download from Hugging Face**

You can manually download the model from:
🔗 [Hugging Face Model Link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

Then, load it locally:

```python
model = SentenceTransformer("/path/to/downloaded/model")
```

#### **3. Verify Installed Model**

To check if it's installed and where it’s stored:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
print(model)
```

Would you like help with using a **custom model** for FAISS indexing instead of this pre-trained one? 🚀
