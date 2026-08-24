import os
from pathlib import Path
from pyexpat import model
from dotenv import load_dotenv
from groq import Groq  

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"
role = "user"
content = "Write a short poem about the beauty of nature."

message = {
    'role': role,
    'content': content
}

message = [message]
response = client.chat.completions.create(model = model ,  messages=message)
print(response) 

answer = response.choices[0].message.content
print(answer)
