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
