
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
    messages = []

    system_content = ""

    if "description" in prompt_config:
        if isinstance(prompt_config["description"], list):
            system_content += "# " + " ".join(prompt_config["description"]) + "\n\n"
        else:
            system_content += "# " + prompt_config["description"] + "\n\n"

    system_content += prompt_config.get("role", "")

    if "personality" in prompt_config:
        system_content += "\n\n" + prompt_config["personality"]

    if "style_or_tone" in prompt_config:
        system_content += "\n\n" + "\n".join(prompt_config["style_or_tone"])

    if "output_constraints" in prompt_config:
        system_content += "\n\n" + "\n".join(prompt_config["output_constraints"])

    if "output_format" in prompt_config:
        system_content += "\n\n" + "\n".join(prompt_config["output_format"])

    messages.append(SystemMessage(content=system_content))

    if "instruction" in prompt_config:
        input_content = f"{prompt_config['instruction']}\n\n{input_data}"
    else:
        input_content = input_data

    messages.append(HumanMessage(content=input_content))

    return messages

