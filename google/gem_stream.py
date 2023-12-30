import os
from dotenv import load_dotenv
from pathlib import Path
import streamlit as st
from tqdm import tqdm
import time

from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.multi_modal_llms.generic_utils import load_image_urls
import google.generativeai as genai

dotenv_path = Path('./.env')
load_dotenv(dotenv_path=dotenv_path)  # add your GOOGLE API key here

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"), transport='rest')

# Initialize GeminiMultiModal
gemini_pro = GeminiMultiModal(model_name="models/gemini-pro-vision")

# Streamlit App
st.title("Product Description Sales App")

# Add an input area for user to input a prompt
prompt = st.text_input("Enter a prompt:", "Identify the jewelry in the image and write a Christmas holiday sale description for it.")

# Add an input area for user to input image URLs
image_urls = st.text_area("Enter image URLs (one per line):", "https://www.estrogine.com/wp-content/uploads/2023/09/81615CCD-7344-41F7-B5A9-245CD3CC8383.jpeg\n")

# Split the input into a list of URLs
image_urls = [url.strip() for url in image_urls.split('\n') if url.strip()]

# Display the images
if image_urls:
    st.subheader("Uploaded Images:")
    st.image(image_urls, caption='Uploaded Images.', use_column_width=True)

# Add a button to trigger the Gemini API call
if st.button("Generate Description") and image_urls:
    # Progress bar
    progress_bar = st.progress(0)
    
    # Load Images from URLs
    image_documents = load_image_urls(image_urls)

    # Perform Gemini API call with tqdm support
    with st.spinner("Generating description..."):
        for _ in tqdm(range(100), desc="Processing", dynamic_ncols=True):
            time.sleep(0.01)  # Simulate some work
            progress_bar.progress(_ / 100)
            
        # Finalizing the Gemini API call
        complete_response = gemini_pro.complete(prompt=prompt, image_documents=image_documents)
        
    # Display the result
    st.subheader("Generated Description:")
    st.write(complete_response)
