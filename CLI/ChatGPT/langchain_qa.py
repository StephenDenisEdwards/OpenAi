#!/usr/bin/env python
import os
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings  # Added import
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

def main():
    print("Current working directory:", os.getcwd())  # Debug: show current working directory
    # Load all documents from the "data" directory
    loader = DirectoryLoader("data")
    persist_dir = "persist"
    # Updated: pass embedding and vectorstore_kwargs instead of persist_directory
    index_creator = VectorstoreIndexCreator(
        embedding=OpenAIEmbeddings(), 
        vectorstore_kwargs={"persist_directory": persist_dir}
    )
    index = index_creator.from_loaders([loader])
    
    # Set up the conversational retrieval chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1})
    )
    
    chat_history = []
    while True:
        query = input("Enter your query (or 'quit' to exit): ")
        if query.lower() in ["quit", "q", "exit"]:
            break
        result = qa_chain({"question": query, "chat_history": chat_history})
        answer = result.get("answer")
        print("\n" + answer + "\n")
        chat_history.append((query, answer))

if __name__ == "__main__":
    main()
