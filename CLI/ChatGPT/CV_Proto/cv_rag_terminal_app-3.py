import os
import openai
import faiss
import numpy as np
import docx2txt
from pathlib import Path
from docx import Document
from typing import List
from openai.embeddings_utils import get_embedding
import PyPDF2  # new import for PDF processing

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip() != ""])

# Load and parse CVs from DOCX files
def extract_text_from_docx_using_docx2txt(file_path: Path) -> str:
    text = docx2txt.process(str(file_path))
    return text

def extract_text_from_pdf(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def chunk_text(text: str, max_tokens: int = 300) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    candidate_name = "Unknown"
    current_chunk = ""
    
    for para in paragraphs:
        if para.startswith("CV:"):
            # Start new candidate section: extract and set candidate name
            if current_chunk:
                chunks.append(current_chunk)
            candidate_name = para[3:].strip() or "Unknown"
            current_chunk = f"Candidate: {candidate_name}\n"
        else:
            # Append paragraph; check token length to avoid overflow
            if len(current_chunk.split()) + len(para.split()) > max_tokens:
                chunks.append(current_chunk)
                current_chunk = f"Candidate: {candidate_name}\n" + para
            else:
                current_chunk += "\n" + para
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def embed_chunks(chunks: List[str]) -> np.ndarray:
    return np.array([get_embedding(chunk, engine="text-embedding-3-small") for chunk in chunks])

# Alternative similarity search options using different FAISS indices
def build_faiss_index(embeddings: np.ndarray, index_type: str = "L2"):
    """
    Build a FAISS index using the specified similarity metric.
    Options for index_type:
        - "L2": Uses Euclidean distance (L2) -- default.
        - "IP": Uses inner product. If embeddings are normalized, this yields cosine similarity.
    """
    dim = embeddings.shape[1]
    if index_type == "L2":
        index = faiss.IndexFlatL2(dim)
    elif index_type == "IP":
        index = faiss.IndexFlatIP(dim)
    else:
        raise ValueError("Unsupported index type. Use 'L2' or 'IP'.")
    index.add(embeddings)
    return index

# Modified answer_query to choose the index type for similarity search
def answer_query(enhanced_query: str, query: str, chunks: List[str],
                    embeddings: np.ndarray, top_k: int = 3, index_type: str = "IP") -> str:
    query_embedding = get_embedding(enhanced_query + query, engine="text-embedding-3-small")
    
    # If using inner product, normalize embeddings and query embedding for cosine similarity search.
    if index_type == "IP":
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        index = build_faiss_index(norm_embeddings, index_type="IP")
        _, indices = index.search(np.array([query_embedding]), top_k)
    else:
        index = build_faiss_index(embeddings, index_type="L2")
        _, indices = index.search(np.array([query_embedding]), top_k)
    
    context = "\n\n".join([chunks[i] for i in indices[0]])
    prompt = f"Based on the following CV excerpts, answer the question using the specified format:\n\n{enhanced_query}\n\n{context}"

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an expert technical recruiter who always outputs the candidate's name using the provided format."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"]
def answer_query(enhanced_query: str, query: str, chunks: List[str], embeddings: np.ndarray, top_k: int = 3) -> str:
    query_embedding = get_embedding(enhanced_query + query, engine="text-embedding-3-small")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    _, indices = index.search(np.array([query_embedding]), top_k)

    context = "\n\n".join([chunks[i] for i in indices[0]])
    # prompt = f"Based on the following CV excerpts, answer the question using the specified format:\n\n{enhanced_query}\n\n{context}"

    # prompt = f"Based on the following CV excerpts, answer the question using the specified format:\n\n{enhanced_query}\n\n{context}"
    prompt = f"Based on the following CV excerpts, answer the question: {query}\n\n{context}"

    #response = openai.ChatCompletion.create(
    #    model="gpt-4-turbo",
    #    messages=[
    #        {"role": "system", "content": "You are an expert technical recruiter who always outputs the candidate's name using the provided format."},
    #        {"role": "user", "content": prompt}
    #    ]
    #)

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
    # Load and process all CVs (PDF and DOCX) in cv_dir
    all_cv_texts = []
    for cv_file in cv_dir.glob("*.*"):

        if cv_file.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(cv_file)
        elif cv_file.suffix.lower() == ".docx":
            text = extract_text_from_docx_using_docx2txt(cv_file)
        else:
            continue
        candidate_name = cv_file.stem  # use file name as candidate name
        
        print(f"Processing {cv_file.name} for candidate: {candidate_name}")

        all_cv_texts.append(f"CV: {candidate_name}\n{text}")
    
    combined_text = "\n\n".join(all_cv_texts)

    chunks = chunk_text(combined_text)
    print(f"Split into {len(chunks)} chunks. Generating embeddings...")

    embeddings = embed_chunks(chunks)
    print("Embeddings generated. Ready to answer questions.")

    # Sample prompts    
    # determine who has the most experience in C# programming. Provide a reasoned answer.
    # who has the most C# experience

    while True:
        choice = input("Enter 'j' to add a job posting or type your search query: ")
        if choice.lower() == "j":
            job_posting = input("Please enter the job posting text: ")
            enhanced_query = """
            """
            # Prepend detailed instruction to always include the candidate's name
            #enhanced_query = """
            #    Answer in the following format without deviation:\n
            #    Candidate: [Candidate Name]\n
            #    Explanation: [Your explanation].\n
            #    If you cannot identify a candidate, put 'Unknown' as the name and provide your reasoning.\n
            #    The Explanation should contain your detailed bullet pointed reasoning.\n
            #"""
            query = f"Which candidate would suit the following position best: {job_posting}"
            # answer = answer_query(enhanced_query, query, chunks, embeddings)
            answer = answer_query(enhanced_query, query, chunks, embeddings, 10)
            print(f"\nComparison Result:\n{answer}\n")
            continue
        query = choice
        if query.lower() in ("exit", "quit"):
            break
        print(f"\nQuery:\n{query}\n")
            # Prepend detailed instruction to always include the candidate's name
        enhanced_query = """" \
        """
        # enhanced_query = (
        #        "Candidate: [Candidate Name]\n"
        #        "Explanation: [Your explanation].\n"
        #        "The Explanation should contain your detailed bullet pointed reasoning.\n"
        #    )
        answer = answer_query(enhanced_query, query, chunks, embeddings)
        print(f"\nAnswer:\n{answer}\n")
