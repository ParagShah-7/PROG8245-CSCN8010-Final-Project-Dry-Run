import torch
from transformers import AutoTokenizer, pipeline
from functools import lru_cache
from better_profanity import profanity
import language_tool_python

# Load profanity words
profanity.load_censor_words()

# Load tokenizer and summarizer once
@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")

@lru_cache(maxsize=1)
def get_summarizer():
    return pipeline("summarization", model="t5-small", device=0 if torch.cuda.is_available() else -1)

# Grammar correction using LanguageTool (Java required)
def correct_prompt(prompt: str) -> str:
    try:
        tool = language_tool_python.LanguageTool('en-US')
        return tool.correct(prompt)
    except Exception as e:
        print(f"[Correction failed] {e}")
        return prompt

# Prompt complexity in tokens
def compute_complexity(prompt):
    return len(get_tokenizer().tokenize(prompt or ""))

# Inference-only energy estimation
def estimate_energy(layers, time_hours, flops_per_hour, complexity, mode="inference"):
    try:
        multiplier = 1e-12  # Inference only
        return round(multiplier * float(layers) * float(time_hours) * float(flops_per_hour) * float(complexity), 6)
    except:
        return 0.0

# Validate prompt (non-empty and clean)
def is_valid_prompt(prompt: str) -> bool:
    return bool(prompt.strip()) and not profanity.contains_profanity(prompt)

# Simplify corrected prompt using T5 summarizer
def generate_multiple_prompts(prompt, count=5):
    summarizer = get_summarizer()
    corrected_prompt = correct_prompt(prompt)
    decode_configs = [
        {"do_sample": False},
        {"do_sample": True, "top_k": 50},
        {"do_sample": True, "top_k": 100, "top_p": 0.92},
        {"do_sample": True, "top_k": 30, "top_p": 0.8},
        {"do_sample": True, "num_beams": 4}
    ]
    results = []
    original_complexity = compute_complexity(corrected_prompt)

    for config in decode_configs[:count]:
        try:
            summary = summarizer(corrected_prompt, max_length=20, min_length=5, **config)
            simplified = summary[0]['summary_text']
            new_complexity = compute_complexity(simplified)
            if new_complexity <= original_complexity:
                energy = estimate_energy(12, 1.0, 12e9, new_complexity)
                results.append({
                    "prompt": simplified,
                    "complexity": new_complexity,
                    "energy": energy,
                    "strategy": str(config)
                })
        except Exception as e:
            print(f"[Simplification error] {e}")
            continue
    return results
