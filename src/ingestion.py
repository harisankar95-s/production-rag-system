import logging
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.config import config
from src.utils import get_embedding_model

logger = logging.getLogger(__name__)

def load_documents(file_path, encoding='utf-8'):
    logger.info(f"Loading documents from: {file_path}")
    documents = TextLoader(file_path, encoding)
    documents = documents.load()
    logger.info(f"Loaded {len(documents)} document(s)")
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

def ingest(file_path):
    documents = load_documents(file_path)
    chunks    = chunk_documents(documents)
    embedding_model = get_embedding_model()
    vectorstore      = build_vectorstore(chunks,embedding_model)
    return vectorstore

