
# modular-rag-assistant
RAG-based, vector storage, corpus embedding, prompt/retrieval/response AI assistant with modular prompt based on user query.
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



## Usage

### Modular prompt chat assistant

- **`python code/main.py`**

This launches the assistant in your terminal or command-line interface.
You’ll be prompted to enter a question about football, neuroscience, or sign language AI research, or type config or exit.





## License

This project is licensed under the [MIT License](LICENSE).
