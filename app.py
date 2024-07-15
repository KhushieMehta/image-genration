import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Define a function to load the model
@st.cache_resource
def load_model(hugging_token):
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=hugging_token)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

# Define a function to generate images
def generate_image(prompt, model):
    with torch.no_grad():
        image = model(prompt).images[0]
    return image

# Streamlit app
st.title("AI Image Generator")
st.write("Enter a prompt to generate an image using Stable Diffusion.")

# User input
hugging_token = st.text_input("Enter your Hugging Face access token:", type="password")
prompt = st.text_input("Enter your image prompt:")

# Load model on button click
if st.button("Generate Image"):
    if hugging_token and prompt:
        with st.spinner("Loading model and generating image..."):
            model = load_model(hugging_token)
            image = generate_image(prompt, model)
            st.image(image, caption="Generated Image", use_column_width=True)
    else:
        st.error("Please provide both a Hugging Face access token and an image prompt.")
