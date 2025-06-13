
# modular-rag-assistant
RAG-based, vector storage, corpus embedding, prompt/retrieval/response AI assistant.
<br>Ready Tensor Agentic AI Developer Certification - Module 1

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
│   ├── embedding_utils.py # Handles text splitting, embedding generation, and interactions with vector databases.
│   ├── llm_service.py     # Manages communication with the Language Model API for text generation.
│   ├── main.py            # The primary script that orchestrates the entire RAG pipeline.
│   └── prompts.py         # Defines the logic for constructing and formatting modular prompts.
├── data/                  # Directory for raw or processed data files.
├── models/                # Directory to store any local machine learning models (e.g., for topic classification or re-ranking).
│   └── .gitkeep           # A placeholder file to ensure this directory is tracked by Git, even when empty.
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

   Create an `.env` file into root directory including your API key:

   ```
   GROQ_API_KEY=your-api-key-here
   ```

   You can get your API key from [Groq](https://console.groq.com/).



## Usage

### Task 1

- **`name of file .py`**

  - Short description and purpose

- **`name of file2.py`**

  - Short description and purpose.




## License

Information about the project's license.

• License Guide: A primer on licenses for ML projects https://app.readytensor.ai/publications/qWBpwY20fqSz
Choose a License: Help picking an open source license  https://choosealicense.com/
