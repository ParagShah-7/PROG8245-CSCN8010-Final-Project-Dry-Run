import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from business_logic.nlp_module import (
    compute_complexity,
    estimate_energy,
    generate_multiple_prompts,
    is_valid_prompt,
)
from business_logic.prediction_module import predict_category

st.set_page_config(page_title="Sustainable AI App", page_icon="⚡", layout="wide")

# === Global CSS ===
st.markdown(
    """
    <style>
    html, body, .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    .block-container {
        padding-top: 2rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4CAF50, #81C784);
        border: none;
        color: white;
        font-weight: bold;
        padding: 0.5em 1.5em;
        border-radius: 10px;
    }
    .stTextArea textarea, .stNumberInput input {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border-radius: 8px;
    }
    .metric-label, .metric-value {
        font-size: 1.1rem;
        font-weight: bold;
    }
    @media (prefers-color-scheme: light) {
        :root {
            --background-color: #ffffff;
            --text-color: #000000;
            --secondary-background-color: #f0f2f6;
        }
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --background-color: #0f1117;
            --text-color: #ffffff;
            --secondary-background-color: #202124;
        }
    }
    /* Transparent erase button styling */
    button[title="Clear prompt"] {
        position: absolute !important;
        bottom: 9.5rem !important;
        right: 2.5rem !important;
        background: transparent !important;
        border: none !important;
        color: gray !important;
        font-size: 1.4rem !important;
        cursor: pointer !important;
        z-index: 100;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# === Header ===
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        "<h1 style='text-align:center;'>⚡ Sustainable Prompt Energy Estimator ⚡</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:gray;'>Analyze & Optimize AI Prompts Based on Energy Consumption</p><hr>",
        unsafe_allow_html=True,
    )

# === Clear-prompt callback & session-state init ===
def clear_prompt():
    st.session_state["prompt_input"] = ""

st.session_state.setdefault("prompt_input", "")

# === Input Form ===
with st.form(key="prompt_form"):
    st.markdown("### ✍️ Enter your prompt below")
    prompt = st.text_area(
        "Prompt",
        key="prompt_input",
        placeholder="Type your prompt here...",
        height=150,
    )

    col4, col5, col6 = st.columns(3)
    with col4:
        layers = st.slider("🧠 LLM Layers", min_value=1, max_value=200, value=12)
    with col5:
        time = st.slider("⏱️ Training Time (hrs)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    with col6:
        flops = st.slider(
            "🔢 FLOPs/hour", min_value=1e6, max_value=20e9, value=12e9, step=1e6, format="%.0f"
        )

    submit = st.form_submit_button("🚀 Submit")
    improve = st.form_submit_button("💡 Improve")

# === Transparent erase button (outside form) ===
st.button(
    "✖️",
    key="erase_btn",
    help="Clear prompt",
    on_click=clear_prompt,
)

# === Last-run session-state defaults ===
st.session_state.setdefault("last_prompt", "")
st.session_state.setdefault("last_inference_energy", 0.0)
st.session_state.setdefault("last_training_energy", 0.0)
st.session_state.setdefault("last_complexity", 0)

# === Submit Logic ===
if submit:
    if not is_valid_prompt(prompt):
        st.warning("⚠️ Please enter a valid, non-offensive prompt.")
    else:
        complexity = compute_complexity(prompt)
        energy_inf = estimate_energy(layers, time, flops, complexity, mode="inference")
        energy_train = estimate_energy(layers, time, flops, complexity, mode="training")

        # store for later
        st.session_state["last_prompt"] = prompt
        st.session_state["last_complexity"] = complexity
        st.session_state["last_inference_energy"] = energy_inf
        st.session_state["last_training_energy"] = energy_train

        category = predict_category(prompt)

        st.success("✅ Prompt analyzed successfully")
        st.markdown("---")
        st.markdown("### 📊 Prompt Stats")
        c1, c2, c3 = st.columns(3)
        c1.metric("🔋 Inference Energy", f"{energy_inf:.6f} kWh")
        c2.metric("🔋 Training Energy", f"{energy_train:.6f} kWh")
        c3.metric("🧮 Complexity", f"{complexity} tokens")

        if energy_inf > 1000:
            st.warning("⚠️ High inference energy usage detected. Consider simplifying your prompt.")

# === Improve Logic ===
if improve:
    if st.session_state["last_prompt"].strip():
        with st.spinner("🔄 Generating optimized prompts..."):
            variants = generate_multiple_prompts(st.session_state["last_prompt"])
        if variants:
            best = min(variants, key=lambda x: x["energy"])
            improved_complexity = compute_complexity(best["prompt"])

            st.markdown("---")
            st.markdown("### 🌿 Best Optimized Prompt")
            st.code(best["prompt"], language="markdown")

            d1, d2, d3 = st.columns(3)
            d1.metric("🔋 Inference Energy", f"{best['energy']:.6f} kWh")
            d2.metric("🔋 Training Energy", f"{st.session_state['last_training_energy']:.6f} kWh")
            d3.metric("🧮 Complexity (Improved)", f"{improved_complexity} tokens")
        else:
            st.warning("⚠️ Unable to generate optimized prompts.")
    else:
        st.warning("⚠️ Submit a prompt first to improve.")

# === Footer ===
st.markdown(
    """
    <hr>
    <center style='color:gray;'>
    Built by<br>Parag Shah • Kapil Bharadwaj • Preetpal Singh • Sem-1 Team 3 © 2025
    </center>
    """,
    unsafe_allow_html=True,
)
