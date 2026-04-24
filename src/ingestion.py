import logging
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.config import config
import unicodedata

logger = logging.getLogger(__name__)

def load_documents(file_path, encoding='utf-8'):
    logger.info(f"Loading documents from: {file_path}")
    documents = PyPDFLoader(file_path)
    documents = documents.load()
    logger.info(f"Loaded {len(documents)} document(s)")
    return documents

def clean_documents(documents):
    logger.info(f"Cleaning {len(documents)} documents")
    for doc in documents:
        doc.page_content = unicodedata.normalize('NFKD', doc.page_content).encode('ascii', 'ignore').decode('ascii')
    logger.info("Documents cleaned")
    return documents

def chunk_documents(documents):
    logger.info(f"Splitting the document of length {len(documents)}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = config.chunk_size,chunk_overlap = config.chunk_overlap)
    text          = text_splitter.split_documents(documents)
    logger.info(f"Split the documents of length {len(documents)} to {len(text)} chunks")
    return text

def build_vectorstore(chunks,embedding_model):
    logger.info("Creating vector store")
    vector_store = Chroma.from_documents(documents = chunks,embedding = embedding_model,persist_directory = config.chroma_persist_dir,collection_name= config.collection_name)
    logger.info(f"Vector store created in {config.chroma_persist_dir}")
    return vector_store

def inspect_chunks(chunks, n=3):
    logger.info(f"Inspecting first {n} chunks out of {len(chunks)}")
    for i, chunk in enumerate(chunks[:n]):
        print(f"\n--- Chunk {i} ---")
        print(f"Type: {type(chunk)}")
        print(f"Attributes: {chunk.__dict__.keys()}")
        print(f"Metadata: {chunk.metadata}")
        print(f"Content length: {len(chunk.page_content)}")
        print(f"Content preview:\n{chunk.page_content[:300]}")


def ingest(file_path,embedding_model):
    documents = load_documents(file_path)
    documents = clean_documents(documents)
    chunks    = chunk_documents(documents)
    # inspect_chunks(chunks)
    vectorstore      = build_vectorstore(chunks,embedding_model)
    return vectorstore

