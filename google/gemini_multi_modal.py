import os
from dotenv import load_dotenv
from pathlib import Path

from llama_index.multi_modal_llms.gemini import GeminiMultiModal

from llama_index.multi_modal_llms.generic_utils import (
    load_image_urls, 
)
import google.generativeai as genai

dotenv_path = Path('./.env')
load_dotenv(dotenv_path=dotenv_path)  # add your GOOGLE API key here

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"),transport='rest')


## Initialize GeminiMultiModal and Load Images from URLs

image_urls = [
    "https://www.estrogine.com/wp-content/uploads/2023/09/81615CCD-7344-41F7-B5A9-245CD3CC8383.jpeg"
    # Add yours here!
]

image_documents = load_image_urls(image_urls)

gemini_pro = GeminiMultiModal(model_name="models/gemini-pro-vision")

complete_response = gemini_pro.complete(
    prompt="Identify the jewellry in the image and write a Christmas holiday sale description for it.",
    image_documents=image_documents,
)

print(complete_response)



