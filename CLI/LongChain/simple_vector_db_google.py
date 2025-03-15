from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader

# Initialize the Vertex AI embeddings.
# Ensure that your Google credentials are correctly configured (via environment variable or credentials file).
embeddings = VertexAIEmbeddings(
    project="your-google-project",
    location="us-central1",          # adjust the location if needed
    model_name="textembedding-gecko"   # or another supported model name
)

# Load data from a Markdown file
filename = "langchain.md"  # Replace with your .md file path
loader = UnstructuredFileLoader(filename)
docs = loader.load()

# Convert documents to a vector database
vector_db = FAISS.from_documents(docs, embeddings)

# Search for similar results
query = "What is LangChain?"
query_embedding = embeddings.embed_query(query)
results = vector_db.similarity_search_by_vector(query_embedding, k=3)

for res in results:
    print(res.page_content)