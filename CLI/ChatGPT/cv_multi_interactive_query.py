from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Load FAISS Index with metadata filtering
vector_store = FAISS.load_local("cv_multi_faiss_index", embeddings, allow_dangerous_deserialization=True)

# Memory for conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# GPT-4o Model
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Function to determine query type (CV-related or Calendar-related)
def query_faiss(query):
    print(f"\n🔍 Searching FAISS database for: '{query}'...\n")

    # Detect if it's a calendar-related question
    #if any(keyword in query.lower() for keyword in ["schedule", "meeting", "available", "interview", "appointment", "event"]):
    #    retriever = vector_store.as_retriever(search_kwargs={"filter": {"type": "calendar"}})
    #else:
    retriever = vector_store.as_retriever()

    retrieved_docs = retriever.get_relevant_documents(query)

    if retrieved_docs:
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        full_prompt = f"Based on the available data, answer the following question: {query}\n\nData:\n{context}"
        response = llm.predict(full_prompt)
    else:
        print("📡 No relevant data found. Asking GPT-4o directly...\n")
        response = llm.predict(query)

    print("\n🤖 AI: ", response, "\n")

# Interactive chat loop
def chat_with_cv():
    print("💬 Ask about CVs or scheduled events! (Type 'exit' to quit)\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        query_faiss(query)

# Run the chatbot
chat_with_cv()
