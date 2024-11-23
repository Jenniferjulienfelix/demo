!pip install torch torchvision transformers pillow openai streamlit
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import openai
import streamlit as st

# Set OpenAI API Key (replace with your key)
openai.api_key = "YOUR_OPENAI_API_KEY"

# Load CLIP model for image recognition
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to process the image and generate a summary
def generate_humorous_summary(image_path, humor_style="sarcastic"):
    # Step 1: Extract image context
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    outputs = clip_model.get_image_features(**inputs)
    image_features = outputs.detach().numpy().flatten()[:5]  # Simplify for prompt
    context = ", ".join(map(str, image_features))
    
    # Step 2: Generate humor using GPT-3
    prompt = (
        f"Based on the description of the image ({context}), write a humorous summary "
        f"in a {humor_style} tone:\n"
    )
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        temperature=0.9,
        n=1
    )
    return response.choices[0].text.strip()

# Streamlit UI
st.title("Humorous Image Summary Generator")
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png"])
humor_style = st.selectbox("Select Humor Style", ["sarcastic", "punny", "dad jokes", "witty"])

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    with st.spinner("Generating Humorous Summary..."):
        summary = generate_humorous_summary(uploaded_image, humor_style)
    st.success("Summary Generated!")
    st.write(f"**Summary:** {summary}")
