import os
from dotenv import load_dotenv
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt

from llama_index.multi_modal_llms.gemini import GeminiMultiModal
import google.generativeai as genai

from llama_index.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.vector_stores import QdrantVectorStore
from llama_index import SimpleDirectoryReader, StorageContext

import qdrant_client
from llama_index.response.notebook_utils import display_source_node
from llama_index.schema import ImageNode

dotenv_path = Path('./.env')
load_dotenv(dotenv_path=dotenv_path)  # add your GOOGLE API key here

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"),transport='rest')

image_paths = []
for img_path in os.listdir("./cars"):
    image_paths.append(str(os.path.join("./cars", img_path)))


def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9), dpi=200)
    for img_path in image_paths:
        if ".jpg" in img_path:
            image = Image.open(img_path)

            plt.subplot(3, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])
            

            images_shown += 1
            if images_shown >= 9:
                break
            
    plt.show()


# plot_images(image_paths)


image_documents = SimpleDirectoryReader(input_files=["./cars/o1.jpg","./cars/t1.jpg","./cars/v1.jpg"]).load_data()

gemini_llm = GeminiMultiModal(api_key=os.getenv("GOOGLE_API_KEY"), 
                              model_name="models/gemini-pro-vision",
                               max_tokens=1500)

response_1 = gemini_llm.complete(
    prompt="Describe the car in the image in detail",
    image_documents=image_documents,
)

print(response_1)


# Create a local Qdrant vector store
client = qdrant_client.QdrantClient(path="qdrant_mm_db")

text_store = QdrantVectorStore(
    client=client, collection_name="text_collection"
)
image_store = QdrantVectorStore(
    client=client, collection_name="image_collection"
)
storage_context = StorageContext.from_defaults(vector_store=text_store,image_store=image_store)

# Create the MultiModal index
documents = SimpleDirectoryReader("./cars").load_data()
index = MultiModalVectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# generate Text retrieval results
retriever_engine = index.as_retriever(
    similarity_top_k=3, image_similarity_top_k=3
)
retrieval_results = retriever_engine.retrieve("Find the Toyotas")



retrieved_image = []
for res_node in retrieval_results:
    if isinstance(res_node.node, ImageNode):
        retrieved_image.append(res_node.node.metadata["file_path"])
    else:
        display_source_node(res_node, source_length=200)

plot_images(retrieved_image)

# generate Text retrieval results
retriever_engine = index.as_retriever(
    similarity_top_k=3, image_similarity_top_k=3
)
retrieval_results = retriever_engine.retrieve("Find the Teslas")

retrieved_image = []
for res_node in retrieval_results:
    if isinstance(res_node.node, ImageNode):
        retrieved_image.append(res_node.node.metadata["file_path"])
    else:
        display_source_node(res_node, source_length=200)

plot_images(retrieved_image)

query_engine = index.as_query_engine(
    similarity_top_k=3, image_similarity_top_k=3
)
response = query_engine.query("Compare the toyotas")
print(response)