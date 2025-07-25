import streamlit as st
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# Load models
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Alternative prompts
ALTERNATIVES = [
    "Summarize this document.",
    "Explain briefly.",
    "Give a short overview.",
    "List main points only.",
    "Rephrase this simply."
]

# Compute token complexity
def compute_complexity(prompt):
    return len(tokenizer.tokenize(prompt))

# Estimate energy (dummy formula)
def estimate_energy(layers, time, flops, complexity):
    return round(0.000001 * layers * time * flops * complexity, 3)

# Suggest simplified prompt
def suggest_optimized_prompt(prompt):
    prompt_vec = embedder.encode(prompt, convert_to_tensor=True)
    alt_vecs = embedder.encode(ALTERNATIVES, convert_to_tensor=True)
    best = util.cos_sim(prompt_vec, alt_vecs).argmax()
    return ALTERNATIVES[best]

# Streamlit UI
st.title("⚡ Sustainable Prompt Energy Estimator")

prompt = st.text_area("Enter your prompt:")
layers = st.number_input("Model layers", min_value=1, value=12)
time = st.number_input("Training time (hours)", min_value=0.1, value=1.0)
flops = st.number_input("FLOPs/hour", min_value=1e6, value=12e9)

if st.button("Estimate Energy"):
    if prompt.strip():
        complexity = compute_complexity(prompt)
        energy = estimate_energy(layers, time, flops, complexity)
        simpler = suggest_optimized_prompt(prompt)

        st.markdown(f"🔢 **Token Complexity:** {complexity}")
        st.markdown(f"⚡ **Estimated Energy:** `{energy} kWh`")
        st.markdown(f"💡 **Simplified Prompt:** _{simpler}_")
    else:
        st.warning("Please enter a prompt first.")
