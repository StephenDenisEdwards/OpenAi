import os
import openai
import faiss
import numpy as np
import docx2txt
from pathlib import Path
from docx import Document
from typing import List
from openai.embeddings_utils import get_embedding

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip() != ""])

# Load and parse CVs from DOCX files
def extract_text_from_docx_using_docx2txt(file_path: Path) -> str:
    text = docx2txt.process(str(file_path))
    return text

def chunk_text(text: str, max_tokens: int = 300) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # Always start a new chunk when encountering a CV header
        if para.startswith("CV:"):
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para
        else:
            # Check if adding the current paragraph exceeds the max token limit
            if len(current_chunk.split()) + len(para.split()) > max_tokens:
                chunks.append(current_chunk)
                current_chunk = para
            else:
                current_chunk += "\n" + para
    
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def embed_chunks(chunks: List[str]) -> np.ndarray:
    return np.array([get_embedding(chunk, engine="text-embedding-3-small") for chunk in chunks])

def answer_query(query: str, chunks: List[str], embeddings: np.ndarray, top_k: int = 3) -> str:
    query_embedding = get_embedding(query, engine="text-embedding-3-small")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    _, indices = index.search(np.array([query_embedding]), top_k)

    context = "\n\n".join([chunks[i] for i in indices[0]])
    prompt = f"Based on the following CV excerpts, answer the question: {query}\n\n{context}"

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an expert technical recruiter."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    cv_dir = base_dir / "cvs"
    # Load and process CVs
    # jim_text = extract_text_from_docx(cv_dir / "Jim Taliadoros CV.docx")
    # stephen_text = extract_text_from_docx(cv_dir / "Stephen Edwards CV.docx")

    jim_text =     extract_text_from_docx_using_docx2txt(cv_dir / "Jim Taliadoros CV.docx")
    stephen_text =     extract_text_from_docx_using_docx2txt(cv_dir / "Stephen Edwards CV.docx")
    combined_text = f"CV: Jim Taliadoros\n{jim_text}\n\nCV: Stephen Edwards\n{stephen_text}"

    chunks = chunk_text(combined_text)
    print(f"Split into {len(chunks)} chunks. Generating embeddings...")

    embeddings = embed_chunks(chunks)
    print("Embeddings generated. Ready to answer questions.")

    # Sample prompts    
    # determine who has the most experience in C# programming. Provide a reasoned answer.
    # who has the most C# experience

    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() in ("exit", "quit"):
            break
        answer = answer_query(query, chunks, embeddings)
        print(f"\nAnswer:\n{answer}\n")
