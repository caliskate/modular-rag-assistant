# Handle UTF-8, get main logger, 
# Define setup_logging, load env vars, main app loop while selecting prompt by similarity
import os
import sys
import io
import logging
from dotenv import load_dotenv # For loading environment variables
from utils import load_yaml_config # For loading YAML configurations
from utils import load_all_publications, get_db_collection, initialize_db, insert_publications # Import functions from utils
from paths import VECTOR_DB_DIR # Import the path constant
from utils import select_prompt_by_similarity
from llm_service import respond_to_query # For handling RAG responses
from paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, OUTPUTS_DIR # For path constants

# Ensure stdout can handle UTF-8 characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.stream.reconfigure(encoding='utf-8')
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # Add handlers to the logger
    if not logger.handlers: # Prevent adding handlers multiple times if main is reloaded
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

## Load environment variables at the application start
load_dotenv()

if __name__ == "__main__":
    setup_logging() # Configure logging
    logger.info("Starting RAG Assistant...")

    # Load application and prompt configurations
    app_config = load_yaml_config(APP_CONFIG_FPATH)
    prompt_config_all = load_yaml_config(PROMPT_CONFIG_FPATH)
        # Initialize LLM model and vector DB defaults from config.yaml
    llm_model_name = app_config.get("DEFAULT_LLM_MODEL_NAME", "llama-3.1-8b-instant") # Config Model and default model
    vectordb_params = {
        "threshold": app_config.get("DEFAULT_SIMILARITY_THRESHOLD", 0.5),
        "n_results": app_config.get("TOP_K_RETRIEVAL", 5),
    }
    
    # Topic names used for similarity matching prompts
    rag_prompts = {
    "football analytics": prompt_config_all["football_prompt"],
    "neuroscience research": prompt_config_all["neuro_prompt"],
    "sign language recognition": prompt_config_all["sign_language_prompt"],
    }

    exit_app = False
    while not exit_app:
        query = input(
            "\nEnter an AI research question about football, neuroscience or sign language, 'config' to change retrieval parameters, or 'exit' to quit: "
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


