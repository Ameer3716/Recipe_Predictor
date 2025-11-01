import streamlit as st
import torch
import os

# Check if running in Streamlit Cloud or locally
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.error("⚠️ The `transformers` library is not installed. Please add it to requirements.txt")
    st.stop()

@st.cache_resource
def load_model():
    """Load the model and tokenizer with caching"""
    # Load from Hugging Face Hub
    MODEL_NAME = "Ameer15/Recipe_predictor"
    
    try:
        st.info(f"Loading model from Hugging Face: {MODEL_NAME}")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
        
        st.success(f"✅ Model loaded successfully on {device.upper()}")
        return tokenizer, model
    
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {str(e)}")
        st.info(f"""
        **Troubleshooting:**
        
        1. Verify the model exists at: https://huggingface.co/{MODEL_NAME}
        2. Check that the model upload completed successfully
        3. Ensure requirements.txt includes: transformers>=4.30.0
        
        If the model isn't uploaded yet, run the upload script in your Kaggle notebook.
        """)
        st.stop()

# Load model
tokenizer, model = load_model()

# UI
st.title("🍳 Recipe Generator")
st.markdown("Generate creative recipes from titles and ingredients using GPT-2!")

# Input fields
col1, col2 = st.columns(2)

with col1:
    title = st.text_input(
        "Recipe Title",
        value="Spicy Chickpea Curry",
        help="Enter a name for your dish"
    )

with col2:
    temperature = st.slider(
        "Creativity (Temperature)",
        min_value=0.5,
        max_value=1.5,
        value=0.9,
        step=0.1,
        help="Higher values = more creative/random"
    )

ings = st.text_area(
    "Ingredients (comma separated)",
    value="chickpeas, onion, garlic, tomato, spices, oil, salt",
    height=100,
    help="List the main ingredients you want to use"
)

# Advanced options
with st.expander("⚙️ Advanced Settings"):
    max_length = st.slider("Maximum Length", 200, 600, 400, 50)
    top_p = st.slider("Top-p (nucleus sampling)", 0.5, 1.0, 0.9, 0.05)

# Generate button
if st.button("🎲 Generate Recipe", type="primary"):
    if not title.strip() or not ings.strip():
        st.warning("Please provide both a title and ingredients!")
    else:
        with st.spinner("Cooking up your recipe... 👨‍🍳"):
            try:
                # Create prompt
                prompt = f"Title: {title}\nIngredients: {ings}\nRecipe:\n"
                
                # Tokenize
                ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
                
                # Generate
                out = model.generate(
                    ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
                # Decode
                txt = tokenizer.decode(out[0], skip_special_tokens=True)
                recipe = txt.split("Recipe:\n")[-1].strip()
                
                # Display result
                st.success("✨ Recipe generated!")
                st.markdown("### Your Recipe:")
                st.text_area(
                    "Instructions",
                    value=recipe,
                    height=300,
                    label_visibility="collapsed"
                )
                
            except Exception as e:
                st.error(f"Error generating recipe: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    Made with ❤️ using GPT-2 fine-tuned on recipe data
</div>
""", unsafe_allow_html=True)
