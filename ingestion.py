import os
import json
import hashlib
from dotenv import load_dotenv
import pymupdf as fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_TOKEN")

HASH_FILE = "doc_hashes.json"
COLLECTION_NAME = "rag_documents_v2"
QDRANT_PATH = "qdrant_storage_v2"
BATCH_SIZE = 50

# Hash functions

def calculate_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_hashes():
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            return json.load(f)
    return {}

def save_hashes(hashes):
    with open(HASH_FILE, "w") as f:
        json.dump(hashes, f, indent=2)

def is_document_changed(file_path, saved_hashes):
    filename = os.path.basename(file_path)
    current_hash = calculate_hash(file_path)
    if filename not in saved_hashes:
        return True, current_hash
    if saved_hashes[filename] != current_hash:
        return True, current_hash
    return False, current_hash

# PDF Files Loading

# def load_pdf(file_path):
#     docs = fitz.open(file_path)
#     text = ""
    # for page in docs:
    #     text += page.get_text("text")
    #     text = text.replace("\n", " ")
    # return text
# def load_pdf(file_path):
#     with fitz.open(file_path) as docs:
#         return " ".join(
#             page.get_text("text").replace("\n", " ")
#             for page in docs)
def load_pdf(file_path):

    text = ""

    with fitz.open(file_path) as doc:

        for page_num, page in enumerate(doc):

            page_text = page.get_text("text")

            # clean text
            page_text = page_text.replace("\n", " ")

            text += f"\n\nPAGE {page_num + 1}\n\n"

            text += page_text

    return text

#chunking
# def create_chunks_with_metadata(text, file_path):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=150,length_function=len)
#     chunks = text_splitter.split_text(text)
#     filename = os.path.basename(file_path)
#     metadatas = [{"source": filename,"chunk_index": i,"total_chunks": len(chunks)}for i, chunk in enumerate(chunks)]
#     return chunks, metadatas
def create_chunks_with_metadata(text, file_path):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1200,
        chunk_overlap=200,

        separators=["\n\n","\n",". ","; ",", "," "])

    chunks = splitter.split_text(text)

    filename = os.path.basename(file_path)

    metadatas = []

    for i, chunk in enumerate(chunks):

        metadata = {"source": filename,"chunk_index": i,"total_chunks": len(chunks)}

        metadatas.append(metadata)

    return chunks, metadatas

#QDRANT setup

def setup_qdrant(embeddings):
    client = QdrantClient(path=QDRANT_PATH)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE))
        print(f"Created new collection: {COLLECTION_NAME}")
    
    vector_store = QdrantVectorStore(client=client,collection_name=COLLECTION_NAME,embedding=embeddings)
    return client, vector_store

# deletion pipeline
def delete_document_chunks(client, filename):
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="metadata.source",match=MatchValue(value=filename))]))
        print(f"Deleted old chunks for: {filename}")
    except Exception as e:
        print(f"Deletion error for {filename}: {e}")

# Batch Processing
def ingest_in_batches(vector_store, chunks, metadatas):
    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        batch_metadatas = metadatas[i:i + BATCH_SIZE]
        vector_store.add_texts(batch_chunks, metadatas=batch_metadatas)
        print(f"Ingested batch {i // BATCH_SIZE + 1} of {(total + BATCH_SIZE - 1) // BATCH_SIZE}")

# Ingestion Pipeline

def ingest_document(file_path, client, vector_store, saved_hashes):
    filename = os.path.basename(file_path)
    changed, current_hash = is_document_changed(file_path, saved_hashes)

    if not changed:
        print(f"No changes detected in {filename}, skipping")
        return saved_hashes

    print(f"Processing {filename}...")

    # Delete old chunks if document existed before
    if filename in saved_hashes:
        print(f"Document changed, deleting old chunks...")
        delete_document_chunks(client, filename)

    # Load and chunk
    text = load_pdf(file_path)
    chunks, metadatas = create_chunks_with_metadata(text, file_path)
    print(f"Created {len(chunks)} chunks for {filename}")

    # Batch ingest
    ingest_in_batches(vector_store, chunks, metadatas)

    # Update hash
    saved_hashes[filename] = current_hash
    print(f"Successfully ingested {filename}")

    return saved_hashes

#main    

if __name__ == "__main__":

    embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2",huggingfacehub_api_token=hf_token)

    client, vector_store = setup_qdrant(embeddings)
    saved_hashes = load_hashes()

    data_folder = "data"
    pdf_files = [os.path.join(data_folder, f)for f in os.listdir(data_folder)if f.endswith(".pdf")]

    print(f"Found {len(pdf_files)} PDF files")

    for file_path in pdf_files:
        saved_hashes = ingest_document(file_path, client, vector_store, saved_hashes)

    save_hashes(saved_hashes)
    print("\nAll documents ingested successfully")
    print(f"Hashes saved to {HASH_FILE}")