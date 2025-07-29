import streamlit as st
import os
import warnings
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import torch

# Suppress warnings and logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load tokenizer and embedder
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

# Load real summarization model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

summarizer = load_summarizer()

# Functions
def compute_complexity(prompt: str) -> int:
    return len(tokenizer.tokenize(prompt))

def estimate_energy(layers, time, flops, complexity) -> float:
    try:
        result = 0.000001 * float(layers) * float(time) * float(flops) * float(complexity)
        return round(result, 3)
    except Exception:
        return 0.0

def suggest_optimized_prompt(prompt: str) -> str:
    try:
        summary = summarizer(prompt, max_length=30, min_length=5, do_sample=False)
        return summary[0]['summary_text']
    except Exception:
        return "Could not generate a simplified version."

# Streamlit UI
st.set_page_config(page_title="Sustainable Prompt Energy Estimator", page_icon="⚡")
st.title("⚡ Sustainable Prompt Energy Estimator")

st.markdown("""
Enter a prompt and click **Submit** to estimate energy consumption based on token complexity.  
Click **Improve** to get an optimized version and compare energy savings.
""")

# Reset function
def reset_prompt():
    st.session_state.prompt = ""

# Inputs
prompt = st.text_area("📝 Prompt", key="prompt", placeholder="Type a long prompt to analyze...")
layers = st.number_input("🧠 Number of LLM Layers", min_value=1, max_value=200, value=12, key="layers_input")
time = st.number_input("⏱ Training Time (hrs)", min_value=0.1, max_value=100.0, value=1.0, key="time_input")
flops = st.number_input("🔢 Estimated FLOPs/hour", min_value=1e6, max_value=1e13, value=12e9, step=1e6, key="flops_input")

# Buttons
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    submit = st.button("🚀 Submit")
with col2:
    improve = st.button("💡 Improve")
with col3:
    reset = st.button("🧹 Erase", on_click=reset_prompt)

# Ensure session state
if "last_prompt" not in st.session_state:
    st.session_state["last_prompt"] = ""
if "last_energy" not in st.session_state:
    st.session_state["last_energy"] = 0.0
if "last_complexity" not in st.session_state:
    st.session_state["last_complexity"] = 0

# On Submit
if submit:
    if prompt.strip():
        try:
            complexity = compute_complexity(prompt)
            energy = estimate_energy(layers, time, flops, complexity)

            st.session_state["last_prompt"] = prompt
            st.session_state["last_energy"] = energy
            st.session_state["last_complexity"] = complexity

            st.markdown(f"### 🔋 **Predicted Energy Consumption:** `{energy:,.3f} kWh`")
            st.markdown(f"🧮 Token Complexity: `{complexity}` tokens")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("⚠️ Please enter a prompt.")

# On Improve
if improve:
    last_prompt = st.session_state.get("last_prompt", "")
    if last_prompt.strip():
        try:
            simplified = suggest_optimized_prompt(last_prompt)
            complexity = compute_complexity(simplified)
            energy = estimate_energy(layers, time, flops, complexity)

            st.markdown("### ✨ **Suggested Simplified Prompt:**")
            st.success(simplified)
            st.markdown(f"🔋 **Improved Energy Estimate:** `{energy:,.3f} kWh`")
            st.markdown(f"🧮 Token Complexity: `{complexity}` tokens")
        except Exception as e:
            st.error(f"❌ Error during improvement: {str(e)}")
    else:
        st.warning("⚠️ Submit a prompt before improving it.")
