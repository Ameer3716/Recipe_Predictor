
import streamlit as st, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/kaggle/working/gpt2-recipes")
model = AutoModelForCausalLM.from_pretrained("/kaggle/working/gpt2-recipes").to("cuda" if torch.cuda.is_available() else "cpu")

st.title("Recipe Generator")
title = st.text_input("Title", "Spicy Chickpea Curry")
ings = st.text_area("Ingredients (comma separated)", "chickpeas, onion, garlic, tomato, spices, oil, salt")

if st.button("Generate"):
    prompt = f"Title: {title}\nIngredients: {ings}\nRecipe:\n"
    ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        ids,
        max_length=400,
        temperature=0.9,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    st.text_area("Recipe", txt.split("Recipe:\n")[-1].strip(), height=400)
