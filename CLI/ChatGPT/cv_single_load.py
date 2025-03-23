import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

print("Current working directory:", os.getcwd())  # Debug: show current working directory
print("API Key", os.getenv("OPENAI_API_KEY"))

# Set your OpenAI API Key
# os.environ["OPENAI_API_KEY"] = "your-api-key"

# Load the CV
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "data", "Stephen Edwards CV December 2024.docx")
loader = UnstructuredFileLoader(file_path)
documents = loader.load()

# Add candidate name to metadata for each document
for doc in documents:
    doc.metadata["candidate_name"] = "Stephen Edwards"

# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Generate embeddings and store in FAISS vector database
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)

# Save vector store
vector_store.save_local("cv_faiss_index")

# Example query with metadata filtering to ensure correct candidate:
# results = vector_store.similarity_search("your query here", filter={"candidate_name": "Stephen Edwards"})
# print(results)
