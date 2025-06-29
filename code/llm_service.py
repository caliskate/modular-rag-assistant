# Initializes logger and ChromaDB
# Defines retrieve_relevant_documents and respond_to_query
import logging
import torch # For device selection in embeddings
from langchain_groq import ChatGroq # LLM interaction
from utils import get_db_collection, embed_documents # For DB interaction and embedding queries
from prompts import build_prompt_from_config # For prompt construction

# Initialize the logger for this module
logger = logging.getLogger(__name__)

# Initialize the ChromaDB collection globally within this service, as it's a shared resource for retrieval
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
    "Relevant documents:\n\n"
    + "\n\n".join(relevant_documents)
    + "\n\nUser's question:\n\n"
    + query
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




