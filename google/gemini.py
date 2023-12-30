from llama_index.llms import ChatMessage, Gemini
import google.generativeai as genai

import os
from dotenv import load_dotenv
from pathlib import Path


dotenv_path = Path('./.env')
load_dotenv(dotenv_path=dotenv_path)  # add your GOOGLE API key here

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"),transport='rest')

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)

## Creating Gemini LLM instance        
llm = Gemini(model="models/gemini-pro")



resp = llm.complete("Write a concise product sales email with a product of your choice \
    Return the outputs in json format in a email subject and body format.")
print(resp)