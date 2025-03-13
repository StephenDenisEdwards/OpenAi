import os
from ics import Calendar
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

print("Current working directory:", os.getcwd())  # Debug: show current working directory
print("API Key", os.getenv("OPENAI_API_KEY"))

# Set OpenAI API Key
# os.environ["OPENAI_API_KEY"] = "your-api-key"

# Define directories/files
cv_directory = "data_cv_files"
calendar_directory = "data_calendar_files"
documents_with_metadata = []

# Load CVs and attach metadata
for filename in os.listdir(cv_directory):
    if filename.endswith(".docx") or filename.endswith(".pdf") or filename.endswith(".txt"):
        file_path = os.path.join(cv_directory, filename)
        candidate_name = filename.replace(" CV", "").replace(".docx", "").replace(".pdf", "").replace(".txt", "").replace("_", " ")

        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()

        for doc in documents:
            doc.metadata = {"type": "cv", "name": candidate_name}
            documents_with_metadata.append(doc)
if(False):
    # Load .ics Calendar Files
    for filename in os.listdir(calendar_directory):
        if filename.endswith(".ics"):
            file_path = os.path.join(calendar_directory, filename)
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:  # Modified to specify encoding and error handling
                calendar = Calendar(f.read())
            for event in calendar.events:
                event_text = f"Event: {event.name}\nDate: {event.begin.date()}\nTime: {event.begin.time()}\nLocation: {event.location}\nDescription: {event.description}"
                calendar_doc = Document(page_content=event_text, metadata={"type": "calendar"})
                documents_with_metadata.append(calendar_doc)

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents_with_metadata)

# Store in FAISS with metadata
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)

# Save FAISS index
vector_store.save_local("cv_multi_faiss_index")
print("✅ FAISS Index Updated with CVs and Calendar Events from .ics files!")
