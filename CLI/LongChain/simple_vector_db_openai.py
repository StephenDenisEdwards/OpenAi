from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# Specify the embedding model (for example, text-embedding-ada-002)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Example documents
docs = [Document(page_content="LangChain makes working with LLMs easier! LangChain makes working with LLMs faster!")]

# Convert to vector database
vector_db = FAISS.from_documents(docs, embeddings)

# Search for similar results
query = "What is LangChain?"
query_embedding = embeddings.embed_query(query)
results = vector_db.similarity_search_by_vector(query_embedding, k=3)

for res in results:
    print(res.page_content)
