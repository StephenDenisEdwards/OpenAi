from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Load the FAISS vector store with safe deserialization
vector_store = FAISS.load_local("cv_faiss_index", embeddings, allow_dangerous_deserialization=True)

# Define memory for conversational context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the Conversational Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model_name="gpt-4", temperature=0),
    retriever=vector_store.as_retriever(),
    memory=memory
)

# Start an interactive chat
def chat_with_cv():
    print("💬 Ask me anything about Stephen Edwards' CV! (Type 'exit' to quit)\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = qa_chain.run({"question": query})
        print("\n🤖 AI: ", response, "\n")

# Run the chatbot
chat_with_cv()
