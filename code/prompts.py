
import yaml
import os
from dotenv import load_dotenv
from code.main import select_prompt_by_similarity
# prompt logicfrom langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv() # loads file you must make called "database.env" which contains "GROQ_API_KEY="
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize the LLM with the Groq API key
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # Fast and capable
    temperature=0.7,
    api_key= groq_api_key
)

# "Defines the logic for constructing and formatting modular prompts."

from langchain_core.messages import HumanMessage, SystemMessage

def build_prompt_from_config(prompt_config: dict, input_data: str) -> list:
    """
    Builds a LangChain prompt (list of messages) from a prompt configuration dictionary.
    This is a placeholder and should be adapted to your specific prompt structuring logic.
    """
    messages = []

    # Add system message if role is defined
    if "role" in prompt_config:
        messages.append(SystemMessage(content=prompt_config["role"]))

    # Add instruction (if any)
    if "instruction" in prompt_config:
        messages.append(HumanMessage(content=prompt_config["instruction"]))

    # Add the user input data
    messages.append(HumanMessage(content=input_data))

    # You might want to add personality, style_or_tone, output_constraints here
    # by appending to the system message or adding separate system/human messages
    # for more complex prompt engineering.

    return messages


# code/llm_service.py
import logging
import torch # For device selection in embeddings
from langchain_groq import ChatGroq # LLM interaction
from code.embedding_utils import get_db_collection, embed_documents # For DB interaction and embedding queries
from code.prompts import build_prompt_from_config # For prompt construction

# Initialize the logger for this module
logger = logging.getLogger(__name__)

# Initialize the ChromaDB collection globally within this service, as it's a shared resource for retrieval
# This assumes the DB has already been ingested/initialized by main.py or a separate script.
collection = get_db_collection(collection_name="publications")

def retrieve_relevant_documents(
    query: str,
    n_results: int = 5,
    threshold: float = 0.3,
) -> list[str]:
    """
    Query the ChromaDB database with a string query to retrieve relevant documents.

    Args:
        query (str): The search query string
        n_results (int): Number of results to return (default: 5)
        threshold (float): Threshold for the cosine similarity score (default: 0.3)

    Returns:
        list[str]: A list of relevant document contents.
    """
    logger.info(f"Retrieving relevant documents for query: '{query}'")
    relevant_results = {
        "ids": [],
        "documents": [],
        "distances": [],
    }

    # Embed the query using the same model used for documents
    logger.info("Embedding query...")
    query_embedding = embed_documents([query])[0] # Get the first (and only) embedding

    logger.info("Querying collection...")
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"],
    )

    logger.info("Filtering results...")
    # Filter documents based on the similarity threshold
    keep_item = [False] * len(results["ids"][0])
    for i, distance in enumerate(results["distances"][0]):
        if distance < threshold:
            keep_item[i] = True

    for i, keep in enumerate(keep_item):
        if keep:
            relevant_results["ids"].append(results["ids"][0][i])
            relevant_results["documents"].append(results["documents"][0][i])
            relevant_results["distances"].append(results["distances"][0][i])

    return relevant_results["documents"]

def respond_to_query(
    prompt_config: dict,
    query: str,
    llm_model_name: str, # Renamed 'llm' to 'llm_model_name' to avoid conflict with local llm variable
    n_results: int = 5,
    threshold: float = 0.3,
) -> str:
    """
    Respond to a user query by retrieving relevant documents from ChromaDB
    and then using an LLM to generate a response.
    """
    relevant_documents = retrieve_relevant_documents(
        query, n_results=n_results, threshold=threshold
    )

    logger.info("-" * 100)
    logger.info("Relevant documents: \n")
    for doc in relevant_documents:
        logger.info(doc)
        logger.info("-" * 100)
    logger.info("")

    logger.info("User's question:")
    logger.info(query)
    logger.info("")
    logger.info("-" * 100)
    logger.info("")

    # Combine relevant documents and user query for the LLM input
    input_data = (
        f"Relevant documents:\n\n{relevant_documents}\n\nUser's question:\n\n{query}"
    )

    # Build the RAG prompt using the prompt configuration
    rag_assistant_prompt = build_prompt_from_config(
        prompt_config, input_data=input_data
    )

    logger.info(f"RAG assistant prompt: {rag_assistant_prompt}")
    logger.info("")

    # Initialize the LLM with the specified model name
    llm = ChatGroq(model=llm_model_name)

    # Invoke the LLM with the constructed prompt
    response = llm.invoke(rag_assistant_prompt)
    return response.content


Python

# code/main.py
import os
import logging
from dotenv import load_dotenv # For loading environment variables
from code.utils import load_yaml_config # For loading YAML configurations
from code.llm_service import respond_to_query # For handling RAG responses
from code.config.paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, OUTPUTS_DIR # For path constants

# Get the main logger instance
logger = logging.getLogger(__name__) # Use __name__ for module-specific logger

def setup_logging():
    """
    Configures logging for the application to both a file and console.
    """
    logger.setLevel(logging.INFO)

    # Ensure OUTPUTS_DIR exists before creating log file
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # File handler for logging to a file
    file_handler = logging.FileHandler(os.path.join(OUTPUTS_DIR, "rag_assistant.log"))
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # Add handlers to the logger
    if not logger.handlers: # Prevent adding handlers multiple times if main is reloaded
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

# Load environment variables at the application start
load_dotenv()

# To avoid tokenizer parallelism warning from huggingface
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    setup_logging() # Configure logging
    logger.info("Starting RAG Assistant...")

    # Load application and prompt configurations
    app_config = load_yaml_config(APP_CONFIG_FPATH)
    prompt_config_all = load_yaml_config(PROMPT_CONFIG_FPATH) # Renamed to avoid confusion
    
    # Extract the RAG assistant specific prompt config
    rag_prompts = prompt_config_all["rag_assistant_prompt"]

    # Extract vector database and LLM parameters from app_config
    vectordb_params = app_config.get("vectordb", {}) # Use .get() for safer access
    llm_model_name = app_config.get("llm", "llama-3.1-8b-instant") # Default LLM model

    exit_app = False
    while not exit_app:
        query = input(
            "\nEnter a question, 'config' to change retrieval parameters, or 'exit' to quit: "
        )
        if query.lower() == "exit": # Use .lower() for case-insensitive comparison
            exit_app = True
            logger.info("Exiting application.")
            break # Use break instead of exit() to allow cleanup if any

        elif query.lower() == "config": # Use .lower() for case-insensitive comparison
            try:
                threshold = float(input("Enter the retrieval threshold (e.g., 0.3): "))
                n_results = int(input("Enter the Top K value (number of documents to retrieve, e.g., 5): "))
                vectordb_params = {
                    "threshold": threshold,
                    "n_results": n_results,
                }
                logger.info(f"Retrieval parameters updated: Threshold={threshold}, Top K={n_results}")
            except ValueError:
                logger.error("Invalid input for config. Please enter numbers for threshold and Top K.")
            continue # Continue to the next loop iteration to ask for query again

# Select the prompt dynamically
        selected_prompt = select_prompt_by_similarity(query, rag_prompts)

        response_content = respond_to_query(
            prompt_config=selected_prompt,
            query=query,
            llm_model_name=llm_model_name,
            **vectordb_params,
        )

        logger.info("-" * 100)
        logger.info("LLM response:")
        logger.info(response_content + "\n\n")