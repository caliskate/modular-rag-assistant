# paths.py

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENV_FPATH = os.path.join(ROOT_DIR, ".env")
CODE_DIR = os.path.join(ROOT_DIR, "code")

# Define paths to configuration files
APP_CONFIG_FPATH = os.path.join(ROOT_DIR, 'code', 'config', 'config.yaml')
PROMPT_CONFIG_FPATH = os.path.join(ROOT_DIR, 'code', 'config', 'prompt_config.yaml')

# Define the main output directory for logs and other generated files
OUTPUTS_DIR = os.path.join(ROOT_DIR, 'output')

# Define the path to the data files
DATA_DIR = os.path.join(ROOT_DIR, "data")

PUBLICATION_FPATH = os.path.join(DATA_DIR, "publication.md")

# Define the path to the vector database storage
VECTOR_DB_DIR = os.path.join(ROOT_DIR, 'output', 'vector_db')

CHAT_HISTORY_DB_FPATH = os.path.join(OUTPUTS_DIR, "chat_history.db")
