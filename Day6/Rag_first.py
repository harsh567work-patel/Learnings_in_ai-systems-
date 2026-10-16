import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in your .env file.")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"

#step 1 knowledge base

knowledge_base = [
    "The population of france is approximately 68 million.",
    "The population of germany is approximately 83 million.",
    "The population of italy is approximately 59 million."
]

# step 2 retrieval function

def retrieve_answer(question):
    question = question.lower()
    if "population of france" in question :
        return knowledge_base[0]
    elif "population of germany" in question :
        return knowledge_base[1]
    elif "population of italy" in question :
        return knowledge_base[2]
    else:
        return "I don't have information about that."

def ask_llm(question):
    context = retrieve_answer(question)
    sys_prompt = f"Answer only based on the context provided: {context}"
    system_message = {
        "role": "system",
        "content": sys_prompt
    }
    message = {
        "role": "user",
        "content": question
    }
    messages = [system_message, message]
    response = client.chat.completions.create(model=model, messages=messages)
    answer = response.choices[0].message.content
    return answer

question = "What is the population of france?"
print(ask_llm(question))

# it is good but it very rigid and not flexible. It can only answer questions about the population of france, germany, and italy. If we want to make it more flexible, we can use a vector database to store the knowledge base and retrieve relevant information based on the question. This way, we can add more information to the knowledge base without changing the code.
