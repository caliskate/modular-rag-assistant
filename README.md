
# Abstract
This project implements a modular Retrieval-Augmented Generation (RAG) assistant that dynamically selects specialized prompts based on semantic similarity to the user’s query. Designed to support AI research in focused domains (football analytics, neuroscience, and sign language recognition) the assistant integrates ChromaDB for vector storage, LangChain for LLM orchestration, and Groq-hosted LLaMA 3.1 for fast inference. It emphasizes prompt structure and constraint enforcement to ensure domain-specific, safety-aligned responses.

<p align="center">
  <img src="modular-prompt-photo.png" alt="Modular Prompt Photo" width="50%" />
</p>

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

Stylistic customization such as markdown formatted responses, domain-specific language (e.g., football coach style speach or neurology lab noises, sign language gestures and emoji themed responses with a personified assistant introduction: 🏈,🧠,✌️).

High-speed inference powered by Groq-hosted LLaMA 3.1 models.

A fallback prompt will redirect the user to which topics can be discussed.

Responses are designed to strictly discuss information from each topic's publication.


# 6. Examples

- ask about football
  

<br>user: what can you tell me about football?
<br>LLM response: 
Hey there, I'm Coach Harris from Lincoln High School in Portland, Oregon. I'm excited to chat with you about football.
We can discuss a wide range of topics related to football analytics. Some possible areas of discussion include:
* **Team Performance Metrics**: We can explore how to measure team performance using metrics like points per game, defensive efficiency, and passing yards per attempt.
* **Player Analysis**: I can help you understand how to analyze player performance using advanced metrics like expected passing yards, rushing yards over expectation, and defensive snap count.
* **Game Strategy**: We can discuss how to develop game-winning strategies using data-driven insights, such as identifying opponent weaknesses and exploiting them.
* **Injury Prevention**: I can provide information on how to use data to prevent injuries, such as identifying high-risk positions and practices, and creating injury-reduction protocols.
* What specific area of football analytics would you like to discuss? (blowing whistle sound)


- ask about neuroscience


<br>user: what can you tell me about neuroscience?
<br>LLM response: 
**Hello, I'm Dr. Rachel Kim, a neuroscientist with over 15 years of experience in the field.**

As a neuroscientist, I'd be happy to share my expertise with you. Based on the provided documents, here's what I can tell you about neuroscience:

* Neuroscience is the scientific study of the structure and function of the nervous system, including the brain, spinal cord, and nerves.
* It is a multidisciplinary field that draws on psychology, biology, chemistry, physics, mathematics, and engineering to understand the mechanisms underlying behavior, cognition, and emotion.
* Neuroscientists study the neural basis of perception, attention, memory, learning, and other higher-order functions.
* Techniques used in neuroscience include neuroimaging (e.g., fMRI, EEG), electrophysiology, and behavioral experiments.

(LOUD LABORATORY NOISES)

Some of the key areas of study in neuroscience include:

* Neuroplasticity: the brain's ability to reorganize itself in response to experience and learning.
* Neurodegenerative diseases: conditions such as Alzheimer's, Parkinson's, and Huntington's, which affect the brain and nervous system.
* Neuropharmacology: the study of how drugs interact with the brain and nervous system.


- ask about sign language

<br>user: what can you tell me about sign language?
<br>LLM response:
**Hello, I'm Dr. Rachel Gardner, a researcher at the University of California, Berkeley, specializing in Sign Language Recognition.**

Given the provided documents, here's what I can tell you about Sign Language:

* **Overview**: Sign Language is a visual language used by Deaf and hard-of-hearing communities to communicate.
* **Types**: There are several types of Sign Languages, including American Sign Language (ASL), British Sign Language (BSL), and Langue des Signes Québécoise (LSQ).
* **Recognition**: Sign Language Recognition involves the use of computer vision and machine learning algorithms to recognize and interpret sign language gestures.
* **Applications**: Sign Language Recognition has various applications, including:
        + Communication aids for Deaf and hard-of-hearing individuals
        + Video conferencing and remote communication systems
        + Sign language learning tools and educational resources
* **Challenges**: Some challenges associated with Sign Language Recognition include:
        + Variability in sign language gestures and handshapes
        + Limited training data and domain awareness
        + Difficulty in handling complex sign language expressions and context-dependent meanings

**✌️**
As I conclude, I make the "stop" gesture with my hand, signaling that our conversation has come to an end.
```

- Asking about irrelevant information within an available topic redirects the user to discuss information related to the publication:

```
user: tell me about football parties
LLM response: I'm not aware of any documents or research related to football parties. I can provide general information on football parties if you'd like, but I won't be able to provide specific or in-depth information on the topic. If you're looking for information on football or football-related topics, I'd be happy to help with that.
```

# 8. Conclusion
This modular RAG assistant demonstrates how controlled, prompt-based architectures can guide general-purpose LLMs to deliver reliable, focused, and engaging outputs in specialized domains. With extensible design and safety constraints embedded in the prompting layer, it serves as a robust foundation for future research or productization in domain-aware assistants.






## License

This project is licensed under the [MIT License](LICENSE).
