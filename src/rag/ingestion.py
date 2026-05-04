import logging
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.config import config
import unicodedata
import os
import json

logger = logging.getLogger(__name__)

def load_documents(folder_path, encoding='utf-8'):
    logger.info(f"Loading documents from: {folder_path}")
    documents = {}
    for file in os.listdir(folder_path):
        if file.endswith('.pdf'):
            doc =PyPDFLoader(os.path.join(folder_path, file))
            doc = doc.load()
            documents[file] = doc
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

def filter_chunks(chunks):
    logger.info('Cleaning chuncks')
    longer_chunks = []
    for chunk in chunks:
        if len(chunk.page_content) >200:
            longer_chunks.append(chunk)
    logger.info(f"Filtered chunks: {len(chunks)} → {len(longer_chunks)} (removed {len(chunks) - len(longer_chunks)} junk chunks)")
    return longer_chunks

def _map_summarize(llm, chunks, batch_size=10):
    partial_summaries = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        content = "\n\n".join([chunk.page_content for chunk in batch])
        prompt = f"""Summarize what the following section of a document is about in one sentence.
        
{content}

One sentence summary:"""
        response = llm.invoke(prompt)
        partial_summaries.append(response.content)
    return partial_summaries

def _reduce_summarize(llm, partial_summaries):
    content = "\n".join(partial_summaries)  
    prompt = f"""Combine these section summaries into one sentence 
describing what the entire document is about:

{content}

One sentence summary:"""
    response = llm.invoke(prompt)
    return response.content  

def generate_summary(llm, chunks, batch_size=10):
    partial_summaries = _map_summarize(llm, chunks, batch_size=batch_size)
    doc_summary       = _reduce_summarize(llm,partial_summaries)
    return doc_summary


def ingest(folder_path, embedding_model, llm):
    documents_dict = load_documents(folder_path)
    
    summaries = {}
    all_chunks = []
    
    for filename, docs in documents_dict.items():
        logger.info(f"Processing: {filename}")
        cleaned = clean_documents(docs)
        chunks = chunk_documents(cleaned)
        chunks = filter_chunks(chunks)
        logger.info(f"Generating summary for: {filename}")
        summaries[filename] = generate_summary(llm, chunks)
        all_chunks.extend(chunks)
        
    summary_path = os.path.join(folder_path, 'document_summaries.json')
    with open(summary_path, 'w') as f:
        json.dump(summaries, f, indent=2)
    logger.info(f"Summaries saved to {summary_path}")
    
    vectorstore = build_vectorstore(all_chunks, embedding_model)
    return vectorstore

