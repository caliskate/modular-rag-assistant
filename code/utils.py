import os
import yaml
from dotenv import load_dotenv
from pathlib import Path
from typing import Union, Optional
import glob
import torch
import chromadb
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from code.config.paths import VECTOR_DB_DIR # Import the VECTOR_DB_DIR constant
from paths import DATA_DIR


def load_publication(publication_external_id="yzN0OCQT7hUS"):
    """Loads the publication markdown file.

    Returns:
        Content of the publication as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
    """
    publication_fpath = Path(os.path.join(DATA_DIR, f"{publication_external_id}.md"))

    # Check if file exists
    if not publication_fpath.exists():
        raise FileNotFoundError(f"Publication file not found: {publication_fpath}")

    # Read and return the file content
    try:
        with open(publication_fpath, "r", encoding="utf-8") as file:
            return file.read()
    except IOError as e:
        raise IOError(f"Error reading publication file: {e}") from e


def load_all_publications(publication_dir: str = DATA_DIR) -> list[str]:
    """Loads all the publication markdown files in the given directory.

    Returns:
        List of publication contents.
    """
    publications = []
    for pub_id in os.listdir(publication_dir):
        if pub_id.endswith(".md"):
            publications.append(load_publication(pub_id.replace(".md", "")))
    return publications


def load_yaml_config(file_path: Union[str, Path]) -> dict:
    """Loads a YAML configuration file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If there's an error parsing YAML.
        IOError: If there's an error reading the file.
    """
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {file_path}")

    # Read and parse the YAML file
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}") from e
    except IOError as e:
        raise IOError(f"Error reading YAML file: {e}") from e


def load_env(api_key_type="GROQ_API_KEY") -> None:
    """Loads environment variables from a .env file and checks for required keys.

    Raises:
        AssertionError: If required keys are missing.
    """
    # Load environment variables from .env file
    load_dotenv(ENV_FPATH, override=True)

    # Check if 'XYZ' has been loaded
    api_key = os.getenv(api_key_type)

    assert (
        api_key
    ), f"Environment variable '{api_key_type}' has not been loaded or is not set in the .env file."


def save_text_to_file(
    text: str, filepath: Union[str, Path], header: Optional[str] = None
) -> None:
    """Saves text content to a file, optionally with a header.

    Args:
        text: The content to write.
        filepath: Destination path for the file.
        header: Optional header text to include at the top.

    Raises:
        IOError: If the file cannot be written.
    """
    try:
        filepath = Path(filepath)

        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            if header:
                f.write(f"# {header}\n")
                f.write("# " + "=" * 60 + "\n\n")
            f.write(text)

    except IOError as e:
        raise IOError(f"Error writing to file {filepath}: {e}") from e
    
def initialize_db(
    persist_directory: str = VECTOR_DB_DIR,
    collection_name: str = "publications",
    delete_existing: bool = False,
) -> chromadb.Collection:
    """
    Initialize a ChromaDB instance and persist it to disk.

    Args:
        persist_directory (str): The directory where ChromaDB will persist data.
                                 Defaults to "./vector_db"
        collection_name (str): The name of the collection to create/get.
                               Defaults to "publications"
        delete_existing (bool): Whether to delete the existing database if it exists.
                                Defaults to False
    Returns:
        chromadb.Collection: The ChromaDB collection instance
    """
    if os.path.exists(persist_directory) and delete_existing:
        shutil.rmtree(persist_directory)

    os.makedirs(persist_directory, exist_ok=True)

    # Initialize ChromaDB client with persistent storage
    client = chromadb.PersistentClient(path=persist_directory)

    # Create or get a collection
    try:
        # Try to get existing collection first
        collection = client.get_collection(name=collection_name)
        print(f"Retrieved existing collection: {collection_name}")
    except Exception:
        # If collection doesn't exist, create it - use cosine distance, lower distance is better
        collection = client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine", # Use cosine distance for semantic search
                "hnsw:batch_size": 10000,
            },
        )
        print(f"Created new collection: {collection_name}")

    print(f"ChromaDB initialized with persistent storage at: {persist_directory}")

    return collection

def get_db_collection(
    persist_directory: str = VECTOR_DB_DIR,
    collection_name: str = "publications",
) -> chromadb.Collection:
    """
    Get a ChromaDB client instance.

    Args:
        persist_directory (str): The directory where ChromaDB persists data
        collection_name (str): The name of the collection to get

    Returns:
        chromadb.PersistentClient: The ChromaDB client instance
    """
    return chromadb.PersistentClient(path=persist_directory).get_collection(
        name=collection_name
    )

def chunk_publication(
    publication: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    """
    Chunk the publication into smaller documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_text(publication)

def embed_documents(documents: list[str]) -> list[list[float]]:
    """
    Embed documents using a model.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )

    embeddings = model.embed_documents(documents)
    return embeddings

def insert_publications(collection: chromadb.Collection, publications: list[str]):
    """
    Insert documents into a ChromaDB collection.

    Args:
        collection (chromadb.Collection): The collection to insert documents into
        publications (list[str]): The publications (full text) to insert

    Returns:
        None
    """
    next_id = collection.count() # Get the current count to generate unique IDs

    for publication in publications:
        chunked_publication = chunk_publication(publication)
        embeddings = embed_documents(chunked_publication)
        
        # Generate unique IDs for each chunk
        ids = list(range(next_id, next_id + len(chunked_publication)))
        ids = [f"document_{id}" for id in ids] # Prefix IDs for clarity
        
        collection.add(
            embeddings=embeddings,
            ids=ids,
            documents=chunked_publication,
        )
        next_id += len(chunked_publication) # Update next_id for the next publication

from code.embedding_utils import initialize_db, insert_publications # Import functions from embedding_utils
from code.utils import load_all_publications # Import the data loading utility
from code.config.paths import VECTOR_DB_DIR # Import the path constant

def main():
    """
    Main function to initialize the database, load publications,
    and insert them into the ChromaDB collection.
    """
    # Initialize the ChromaDB collection, deleting existing data if specified
    collection = initialize_db(
        persist_directory=VECTOR_DB_DIR,
        collection_name="publications",
        delete_existing=True, # Set to True to re-create the DB each run for fresh data
    )
    
    # Load all publications from the data directory
    publications = load_all_publications()
    
    # Insert the loaded publications (and their chunks/embeddings) into the database
    insert_publications(collection, publications)

    # Print the total number of documents (chunks) in the collection
    print(f"Total documents (chunks) in collection: {collection.count()}")

if __name__ == "__main__":
    # This block ensures that main() is called only when the script is executed directly
    main()