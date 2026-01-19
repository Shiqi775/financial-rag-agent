import os
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
import chromadb
from tqdm import tqdm
from typing import List
import re

from config import PROCESSED_DATA_DIR, VECTOR_STORE_DIR, EMBED_MODEL_NAME

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: take all the documents from PROCESSED_DATA_DIR, chunk them and
# store them in a vector database with a corresponding vector store index.
# This requires implementing get_all_nodes and get_file_chunks for
# chunking the documents and filling out the 'main'.

# We suggest sticking to the provided template for we believe it to be
# the simplest implementation way. Please, provide explanation if you
# find it necessary to change template.

# Get the embedding model for the vector store index given EMBED_MODEL_NAME.

# Initialize embedding model
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

def get_file_chunks(file_dir: str) -> list[str] | list[TextNode]:
    """
    Given a {ticker}_{year} directory, this method should read the file in there
    and chunk it. The choice of chunking strategy is a part of this task. The most
    basic chunking (using SentenceSplitter) is valued 10 points. More advanced methods
    are valued up to 5 points.

    Splits a file's content into smaller logical chunks using a fixed window with overlap.
    
    Args:
        file_dir (str): The path to the file.
        chunk_size (int): Number of tokens per chunk.
        overlap (int): Number of overlapping tokens between consecutive chunks.

    Returns:
        list[TextNode]: List of chunks as TextNode instances.

    """
    # Read content.txt
    content_path = os.path.join(file_dir, "content.txt")
    with open(content_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Step 1: Split content into sentences
    sentence_splitter = SentenceSplitter()  
    sentences = sentence_splitter.split_text(content)
    
    # Step 2: Create fixed-window chunks with overlap
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If adding this sentence exceeds chunk_size
        if current_size + sentence_length > 512 and current_chunk:
            # Save current chunk
            chunks.append(" ".join(current_chunk))
            
            # Start new chunk with overlap
            # Find sentences to keep for overlap
            overlap_size = 0
            overlap_sentences = []
            # Collect sentences for overlap from the end of the current chunk
            for s in reversed(current_chunk):
                overlap_size += len(s)
                if overlap_size > 50:
                    break
                overlap_sentences.insert(0, s)
                
            # Initialize new chunk with overlap sentences
            current_chunk = overlap_sentences
            current_size = sum(len(s) for s in current_chunk)
        
        # Add the current sentence to the chunk
        current_chunk.append(sentence)
        current_size += sentence_length
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Step 3: Create TextNodes from the chunks
    return [TextNode(text=chunk) for chunk in chunks]


def get_all_nodes(filings_dir: str) -> list[TextNode]:
    """
    Given the filings directory, it should go over the {ticker}_{year} directories
    and get chunks for each of them using get_file_chunks methods, followed by
    creating TextNode instances for each chunk (if they are not already created --
    that will depend on the get_file_chunks implementation) and adding metadata 
    as dictionary: {'ticker': [ticker], 'year': [year]}.

    Process all ticker_year directories and create nodes with metadata.

    Returns:
        list[TextNode]: List of all nodes with metadata.
    """
    all_nodes = []

    # Get all ticker_year directories
    dir_contents = os.listdir(filings_dir)
    
    for dir_name in tqdm(dir_contents, desc="Processing directories"):
        dir_path = os.path.join(filings_dir, dir_name)
        
        # Extract ticker and year from directory name
        ticker, year = dir_name.split('_')
        
        print(dir_path)

        # Get chunks for this filing
        nodes = get_file_chunks(dir_path)
        
        # Add metadata to each chunk
        for node in nodes:
            node.metadata = {
                'ticker': ticker,
                'year': year
            }
            
        # Extend the all_nodes list with the current nodes
        all_nodes.extend(nodes)
        logger.info(f"Processed {dir_name}: {len(nodes)} chunks")

    logger.info(f"Total nodes created: {len(all_nodes)}")
    return all_nodes     

if __name__ == '__main__':
    # TODO: initialize a database which stores all_nodes in VECTOR_STORE_DIR
    # and a corresponding vector store index. We recommend ChromaDB.

    # Create VECTOR_STORE_DIR if it doesn't exist
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
     
    print("Getting nodes from processed files...")
    all_nodes = get_all_nodes(PROCESSED_DATA_DIR)
    print(f"Total nodes created: {len(all_nodes)}")

    # Initialize ChromaDB
    print("Initializing ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    
    # Delete existing collection if it exists
    try:
        chroma_client.delete_collection("financial_filings")
    except Exception as e:
        print(f"Error deleting collection: {e}")
        
    collection = chroma_client.create_collection("financial_filings")
    
    # Create vector store
    print("Creating vector store...")
    vector_store = ChromaVectorStore(chroma_collection=collection, embed_model=embed_model)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create vector store index
    print("Creating vector store index...")
    vector_index = VectorStoreIndex(
        nodes=all_nodes,
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    # Validate stored documents
    collection_data = collection.get()
    print(f"\nTotal documents stored: {len(collection_data['documents'])}")
    for doc, metadata in zip(collection_data['documents'][:3], collection_data['metadatas'][:3]):
        print(f"\nStored Document Preview: {doc[:200]}...")
        print(f"Metadata: {metadata}")

    print("\nVector store construction completed!")


# Possible scores:
# [15 pts]       Chunks are obtained using SentenceSplitter and stored
#                in a vector database in VECTOR_STORE_DIR.
# [<15 pts]      Some mistakes exist (full documents are store in the
#                vector database, the directory is wrong, etc.).
# [up to +5 pts] An advanced chunking method is applied
#                (better than SentenceSplitter).
