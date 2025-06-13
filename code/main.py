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