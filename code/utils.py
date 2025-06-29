# Defines load_publication, load_all_publications, load_yaml_config, load_env, save_text_to_file
# Defines initialize_db, get_db_collection, chunk_publication, embed_documents, insert_publications
import os
# Avoid tokenizer and telemetry warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
import yaml
import logging
import re
from dotenv import load_dotenv
from pathlib import Path
from typing import Union, Optional
from chromadb.config import Settings
import glob
import torch
import chromadb
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from paths import VECTOR_DB_DIR # Import the VECTOR_DB_DIR constant
from paths import DATA_DIR
from sentence_transformers import SentenceTransformer, util

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
    publications = []
    for pub_id in os.listdir(publication_dir):
        if pub_id.endswith(".md"):
            pub_path = os.path.join(publication_dir, pub_id)
            with open(pub_path, "r", encoding="utf-8") as f:
                publications.append(f.read())
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
    This function is good for explicitly managing the DB lifecycle (e.g., clearing).
    """
    if os.path.exists(persist_directory) and delete_existing:
        print(f"Deleting existing ChromaDB at: {persist_directory}")
        shutil.rmtree(persist_directory)

    os.makedirs(persist_directory, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_directory)

    # Use get_or_create_collection here for consistency and robustness
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine", # Use cosine distance for semantic search
            "hnsw:batch_size": 10000,
        },
    )
    print(f"ChromaDB collection '{collection_name}' initialized (get or created) at: {persist_directory}")

    return collection


def get_db_collection(
    persist_directory: str = VECTOR_DB_DIR,
    collection_name: str = "publications",
) -> chromadb.Collection:
    """
    Get a ChromaDB collection. If the collection does not exist, it creates it.
    This function is intended for general access where existence is not guaranteed.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Use get_or_create_collection
    collection = client.get_or_create_collection(
        name=collection_name,
    )
    
    print(f"ChromaDB: Successfully got or created collection '{collection_name}' in '{persist_directory}'")
    return collection

# Chunking function based on markdown headers
def chunk_publication(
    markdown_text: str,
    min_chunk_size: int = 500,
    max_chunk_size: int = 1500,
) -> list[str]:
    """
    Splits markdown content into chunks based on headers (#, ##, ###, etc.).

    Args:
        markdown_text (str): Full markdown text.
        min_chunk_size (int): Minimum chunk size in characters.
        max_chunk_size (int): Maximum chunk size in characters.

    Returns:
        List[str]: List of markdown chunks.
    """
    # Regex pattern to split on markdown headers (keep the header line with chunk)
    # This splits at each line starting with one or more '#'
    pattern = re.compile(r'^(#+ .*)$', re.MULTILINE)

    # Find all headers and split positions
    splits = [(m.start(), m.group(1)) for m in pattern.finditer(markdown_text)]

    # If no headers found, return full text as single chunk
    if not splits:
        return [markdown_text]

    chunks = []
    for i, (start_idx, header) in enumerate(splits):
        # End index is next header start or end of text
        end_idx = splits[i + 1][0] if i + 1 < len(splits) else len(markdown_text)
        chunk = markdown_text[start_idx:end_idx].strip()

        # Merge smaller chunks with previous if below min_chunk_size
        if chunks and len(chunk) < min_chunk_size:
            chunks[-1] += "\n\n" + chunk
        else:
            chunks.append(chunk)


    return chunks

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

logger = logging.getLogger(__name__)

def select_prompt_by_similarity(query: str, topic_prompts: dict, threshold: float = 0.5) -> dict:
    """
    Selects the most relevant prompt based on query similarity to topic names.
    Returns a default prompt if similarity is too low.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    topic_names = list(topic_prompts.keys())
    topic_embeddings = model.encode(topic_names, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)

    similarity_scores = util.pytorch_cos_sim(query_embedding, topic_embeddings)[0]
    best_match_idx = similarity_scores.argmax().item()
    best_score = similarity_scores[best_match_idx].item()

    if best_score < threshold:
        # Fallback generic assistant behavior   
        return {
            "system_message": "You are a friendly assistant that sticks to specific topics only. Answer clearly and concisely.",
            "instruction": "Make sure the user is asking about football, neuroscience, or sign language AI research and nothing else.",
        }

    best_topic = topic_names[best_match_idx]
    return topic_prompts[best_topic]