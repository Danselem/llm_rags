import os
from dotenv import load_dotenv
from pathlib import Path
import streamlit as st
from tqdm import tqdm
import time
from PIL import Image

from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.multi_modal_llms.generic_utils import load_image_urls, ImageDocument
import google.generativeai as genai

dotenv_path = Path('./.env')
load_dotenv(dotenv_path=dotenv_path)  # add your GOOGLE API key here

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"), transport='rest')

# Configure Streamlit theme and layout
st.set_page_config(
    page_title="Product Description App",
    page_icon="✨",
    layout="wide",
)

# Custom styles
custom_styles = """
    .st-bf {
        background-color: #f0f0f0;
        padding: 1rem;
    }
    .st-bo {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0px 0px 10px #888888;
    }
"""

st.markdown(f'<style>{custom_styles}</style>', unsafe_allow_html=True)

# Custom title and header styles
st.markdown(
    """
    <style>
        .title {
            color: #009688;
            font-size: 2.5rem;
            text-align: center;
        }
        .header {
            background-color: #009688;
            color: #ffffff;
            padding: 1rem;
            border-radius: 10px;
        }
        .generated-description {
            background-color: #dff0d8;
            padding: 1rem;
            border-radius: 10px;
        }
        .prompt-section {
            background-color: #d9edf7;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .sidebar-section {
            background-color: #f8d7da;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .button-section {
            margin-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize GeminiMultiModal
gemini_pro = GeminiMultiModal(model_name="models/gemini-pro-vision")

# Streamlit App
st.title("🌟 Product Description App")
st.markdown("<p class='title'>An app for generating beautiful product descriptions!</p>", unsafe_allow_html=True)

# Add an input area for the user to input a prompt in the main page
st.markdown("<div class='prompt-section'><p class='title'>Prompt:</p></div>", unsafe_allow_html=True)
prompt = st.text_input("", "Identify the jewelry in the image and write a Christmas holiday sale description for it.")

# Sidebar
st.sidebar.markdown("<div class='sidebar-section'><p class='title'>Add Image</p></div>", unsafe_allow_html=True)
image_urls = st.sidebar.text_area("Enter image URLs (one per line):", "https://www.estrogine.com/wp-content/uploads/2023/09/81615CCD-7344-41F7-B5A9-245CD3CC8383.jpeg\n")
print(image_urls, type(image_urls))
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Display the images with reduced size if available
if image_urls is not None:
    st.subheader("Attached Image from URL:")
    st.image(image_urls, caption='Image from URL', 
                use_column_width=False, width=500)

elif uploaded_file is not None:
    st.subheader("Uploaded Image:")
    st.image(uploaded_file, caption='Uploaded Image.', 
             use_column_width=False, width=500)


# Load Images from URLs or uploaded file
# image_documents = []
if image_urls is not None:
    image_documents = load_image_urls([image_urls])
    
elif uploaded_file is not None:
    image_documents = ImageDocument(image_path=uploaded_file, caption='Uploaded Image.')
    # image_documents.append(uploaded_image)


# Add a button to trigger the Gemini API call
# submit=st.button("Tell me about the image")
# if submit:
if st.button("Generate Description") and image_documents:
    # Progress bar
    progress_bar = st.progress(0)
    
    # Perform Gemini API call with tqdm support
    with st.spinner("Generating description..."):
        for _ in tqdm(range(100), desc="Processing", dynamic_ncols=True):
            time.sleep(0.01)  # Simulate some work
            progress_bar.progress(_ / 100)
            
        # Finalizing the Gemini API call
        # print(image_documents)
        
        complete_response = gemini_pro.complete(prompt=prompt, image_documents=image_documents)
        
    # Display the result
    st.markdown("<div class='generated-description'><p class='title'>Generated Description:</p></div>", unsafe_allow_html=True)
    st.write(complete_response)
