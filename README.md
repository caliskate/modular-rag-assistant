
# modular-rag-assistant
RAG-based, vector storage, corpus embedding, prompt/retrieval/response AI assistant.
<br>Ready Tensor Agentic AI Developer Certification - Module 1

## Overview

```
modular-rag-assistant/
├── .env                 # Stores sensitive environment variables (e.g., API keys). Not committed to Git.
├── .env.example         # Provides a template for the .env file, showing required variables. Committed to Git.
├── .gitignore           # Specifies intentionally untracked files that Git should ignore.
├── config/
│   ├── config.yaml          # Main application configuration, including reasoning strategies.
│   └── prompt_config.yaml   # Specific configurations for prompt examples, potentially overriding or extending main prompts.
├── data/
│   └── documents/       # Contains all publication documents for RAG.
│       ├── financial_report_q1_2024.md
│       ├── tech_whitepaper_ai.pdf
│       └── medical_journal_article.txt
├── models/              # Directory to store any local machine learning models (e.g., fine-tuned LLMs, topic classifiers).
│   └── .gitkeep         # A placeholder file to ensure Git tracks this empty directory.
├── prompts.py           # Defines all modular prompt components (system, user, topic-specific elements).
├── embedding_utils.py   # Handles text splitting, embedding generation, and vector database operations.
├── llm_service.py       # Manages interactions with the Language Model API for text generation.
├── main.py              # The primary script that orchestrates the entire RAG pipeline.
├── requirements.txt     # Lists all Python dependencies required for the project.
└── README.md            # Provides a general overview, setup instructions, and usage guidelines for the project.
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
