import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from business_logic.nlp_module import compute_complexity, estimate_energy, generate_multiple_prompts
from business_logic.prediction_module import predict_energy, is_anomaly, predict_category

st.set_page_config(page_title="Sustainable AI App", page_icon="⚡", layout="wide")

# === Fixed dark theme ===
primary_color = "#4CAF50"
background_color = "#0f1117"
text_color = "#ffffff"
secondary_bg = "#202124"

st.markdown(f"""
    <style>
    html, body, .main {{ background-color: {background_color}; color: {text_color}; }}
    .block-container {{ padding-top: 2rem; }}
    .stButton > button {{ background: linear-gradient(135deg, {primary_color}, #81C784); border: none; color: white; font-weight: bold; padding: 0.5em 1.5em; border-radius: 10px; }}
    .stTextArea textarea, .stNumberInput input {{ background-color: {secondary_bg}; color: {text_color}; border-radius: 8px; }}
    .metric-label, .metric-value {{ font-size: 1.1rem; font-weight: bold; }}
    </style>
""", unsafe_allow_html=True)

# === Header ===
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(f"<h1 style='text-align:center; color:{text_color};'>⚡ Sustainable Prompt Energy Estimator ⚡</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; color:gray;'>Analyze & Optimize AI Prompts Based on Energy Consumption</p><hr>", unsafe_allow_html=True)

# === User Input Form ===
with st.form(key="prompt_form"):
    st.markdown("### ✍️ Enter your prompt below")
    prompt = st.text_area("Prompt", placeholder="Type your prompt here...", height=150)
    col4, col5, col6 = st.columns(3)
    with col4:
        layers = st.number_input("🧠 LLM Layers", min_value=1, max_value=200, value=12)
    with col5:
        time = st.number_input("⏱️ Training Time (hrs)", min_value=0.1, value=1.0)
    with col6:
        flops = st.number_input("🔢 FLOPs/hour", min_value=1e6, value=12e9, step=1e6, format="%.0f")

    submit_btn, improve_btn = st.columns([1,1])
    with submit_btn:
        submit = st.form_submit_button("🚀 Submit")
    with improve_btn:
        improve = st.form_submit_button("💡 Improve")

# === Session State ===
st.session_state.setdefault("last_prompt", "")
st.session_state.setdefault("last_energy", 0.0)
st.session_state.setdefault("last_complexity", 0)

# === Submit Logic ===
if submit:
    if prompt.strip():
        complexity = compute_complexity(prompt)
        energy = predict_energy(layers, time, complexity)
        category = predict_category(prompt)
        anomaly_flag = is_anomaly(energy)

        st.session_state["last_prompt"] = prompt
        st.session_state["last_energy"] = energy
        st.session_state["last_complexity"] = complexity

        st.success("✅ Prompt analyzed successfully")
        st.markdown("---")
        st.markdown("### 📊 Prompt Stats")
        col7, col8, col9 = st.columns(3)
        col7.metric("🔋 Energy", f"{energy:.3f} kWh")
        col8.metric("🧮 Complexity", f"{complexity} tokens")
        col9.metric("🔮 Category", category)

        if anomaly_flag:
            st.warning("⚠️ Detected as an energy anomaly.")
    else:
        st.warning("⚠️ Please enter a valid prompt.")

# === Improve Logic ===
if improve:
    if st.session_state.get("last_prompt", "").strip():
        with st.spinner("🔄 Generating optimal prompt..."):
            variants = generate_multiple_prompts(st.session_state["last_prompt"])
        if variants:
            best = min(variants, key=lambda x: x["energy"])
            st.markdown("---")
            st.markdown("### 🌿 Best Optimized Prompt")
            st.code(best["prompt"], language="markdown")
            st.metric("🔋 Estimated Energy", f"{best['energy']} kWh")
            st.metric("🧮 Token Complexity", best["complexity"])
        else:
            st.warning("⚠️ Unable to generate optimized prompts.")
    else:
        st.warning("⚠️ Submit a prompt first to improve.")

# === Footer ===
st.markdown("""
<hr>
<center style='color:gray;'>
Built by<br>Parag Shah<br>Kapil Bharadwaj<br>Preetpal Singh<br>Sem-1 Team 3 © 2025
</center>
""", unsafe_allow_html=True)
