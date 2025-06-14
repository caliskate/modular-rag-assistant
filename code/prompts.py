
import yaml
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from utils import select_prompt_by_similarity

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


