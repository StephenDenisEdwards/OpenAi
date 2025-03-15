from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI


# Specify the embedding model (for example, text-embedding-ada-002)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Load data from a Markdown file instead of inline text
filename = "langchain.md"  # Replace with the path to your .md file
loader = UnstructuredFileLoader(filename)
docs = loader.load()


# Split the document into smaller chunks    
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# Convert to vector database
vector_store = FAISS.from_documents(chunks, embeddings)


# Save vector store
# vector_store.save_local("cv_faiss_index")


# Search for similar results
query = "What is LangChain?"
query_embedding = embeddings.embed_query(query)
results = vector_store.similarity_search_by_vector(query_embedding, k=3)

print("-------------------------------------------------------------\n")
for res in results:
    print("\n🤖 Vector DB: ", res.page_content, "\n")

# Add in the LLM into the mix

# Define memory for conversational context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the Conversational Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model_name="gpt-4", temperature=0),
    retriever=vector_store.as_retriever(),
    memory=memory
)

response = qa_chain.run({"question": query})
print("-------------------------------------------------------------\n")
print("\n🤖 AI: ", response, "\n")
print("-------------------------------------------------------------\n")

# Start an interactive chat
def chat_with_cv():
    print("💬 Ask me anything about LangChain! (Type 'exit' to quit)\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = qa_chain.run({"question": query})
        print("\n🤖 AI: ", response, "\n")

chat_with_cv()
