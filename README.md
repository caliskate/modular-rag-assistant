
# Abstract
This project implements a modular Retrieval-Augmented Generation (RAG) assistant that dynamically selects specialized prompts based on semantic similarity to the user’s query. Designed to support AI research in focused domains—such as football analytics, neuroscience, and sign language recognition—the assistant integrates ChromaDB for vector storage, LangChain for LLM orchestration, and Groq-hosted LLaMA 3.1 for fast inference. It emphasizes prompt structure and constraint enforcement to ensure domain-specific, safety-aligned responses.

# 1. Introduction
The rise of domain-specific AI research requires intelligent assistants capable of retrieving and contextualizing relevant publications. This assistant addresses that need by combining modular prompt engineering with vector similarity search, ensuring responses are accurate, scoped, and stylistically aligned with the user’s intent and the associated research domain.

# 2. Dataset

Datasets are in markdown format and are AI research publications related to football, neuroscience and sign language.

# 3. Methodology
The system employs SentenceTransformer embeddings to index and search markdown-based research documents. A user query is embedded and compared with predefined topic categories to select a matching prompt template. Retrieved markdown format chunks based on the relevant documents are then passed, along with the query, into a structured prompt to the LLM. The pipeline leverages OpenAI embeddings for document encoding and LangChain's Groq integration for real-time generation.





## Overview

```
modular-rag-assistant/
├── .env                   # Stores sensitive environment variables (e.g., API keys). Not committed to Git.
├── .env.example           # Provides a template for the .env file, showing required variables. Committed to Git.
├── .gitignore             # Specifies intentionally untracked files and directories that Git should ignore.
├── code/                  # Contains all Python source code for the application.
│   ├── config/            # Configuration files for the application.
│   │   ├── config.yaml          # Main application configuration, including high-level settings and reasoning strategies.
│   │   └── prompt_config.yaml   # Specific configurations for prompt examples or reusable prompt components.
│   ├── paths.py           # Defines important directory and file paths for the project.
│   ├── utils.py           # Handles text splitting, embedding generation, and interactions with vector databases.
│   ├── llm_service.py     # Manages communication with the Language Model API, including retrieval and response generation.
│   ├── main.py            # The primary script that orchestrates the entire RAG pipeline.
│   └── prompts.py         # Defines the logic for constructing and formatting modular prompts.
├── data/                  # Directory for raw or processed data files.
│   ├── football_analytics.md  # Data file for football analytics
│   ├── neuro_persona.md       # Data file for neuro persona research
│   └── sign_language_recognition.md # Data file for sign language recognition
├── models/                # Directory to store any local machine learning models.
│   └── .gitkeep           # Directory placeholder file for Git tracking.
├── output/                # Main output folder for generated files and persistent data.
│   └── vector_db/         # Subfolder to store the local vector database index and associated data.
├── requirements.txt       # Lists all Python package dependencies required to run the project.
└── README.md              # Provides a general overview, setup instructions, and usage guidelines for the project.
```

## Installation


1. **Clone the repository:**

   ```bash
   git clone https://github.com/caliskate/modular-rag-assistant.git
   cd modular-rag-assistant
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Groq API key:**

   Create an `.env` file into root directory and add your API key:

   ```
   GROQ_API_KEY=your-api-key-here
   ```

   You can get your API key from [Groq](https://console.groq.com/).

# 4. How it Works
When a user enters a query, the assistant computes the semantic similarity of the query to predefined topics using all-MiniLM-L6-v2. Based on the best match, a domain-specific prompt (e.g., for football, neuroscience, or sign language) is selected. The assistant then queries a ChromaDB vector store for relevant documents, constructs a rich LLM prompt using these documents and the selected template, and finally returns a formatted response from the Groq LLaMA model.

## Usage

### Modular prompt chat assistant

- **`python code/main.py`**

This launches the assistant in your terminal or command-line interface.
You’ll be prompted to enter a question about football, neuroscience, or sign language AI research, or type config or exit.


# 5. Key Features
Modular prompt selection via query-topic similarity.

Document-grounded responses using ChromaDB vector search.

Domain enforcement through strict output constraints in prompt configuration.

Stylistic customization such as markdown formatting and domain-specific language (e.g., coach-style speech or lab noises).

High-speed inference powered by Groq-hosted LLaMA 3.1 models.

# 6. Examples


# 8. Conclusion
This modular RAG assistant demonstrates how controlled, prompt-based architectures can guide general-purpose LLMs to deliver reliable, focused, and engaging outputs in specialized domains. With extensible design and safety constraints embedded in the prompting layer, it serves as a robust foundation for future research or productization in domain-aware assistants.






## License

This project is licensed under the [MIT License](LICENSE).
