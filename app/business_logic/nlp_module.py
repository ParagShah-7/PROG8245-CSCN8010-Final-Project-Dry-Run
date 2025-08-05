import torch
from transformers import AutoTokenizer, pipeline
from functools import lru_cache

@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")

@lru_cache(maxsize=1)
def get_summarizer():
    return pipeline("summarization", model="t5-small", device=0 if torch.cuda.is_available() else -1)

def compute_complexity(prompt):
    return len(get_tokenizer().tokenize(prompt or ""))

def estimate_energy(layers, time_hours, flops_per_hour, complexity):
    try:
        return round(1e-6 * float(layers) * float(time_hours) * float(flops_per_hour) * float(complexity), 3)
    except:
        return 0.0

def generate_multiple_prompts(prompt, count=5):
    summarizer = get_summarizer()
    decode_configs = [
        {"do_sample": False},
        {"do_sample": True, "top_k": 50},
        {"do_sample": True, "top_k": 100, "top_p": 0.92},
        {"do_sample": True, "top_k": 30, "top_p": 0.8},
        {"do_sample": True, "num_beams": 4}
    ]
    results = []
    for config in decode_configs[:count]:
        try:
            summary = summarizer(prompt, max_length=20, min_length=5, **config)
            simplified = summary[0]['summary_text']
            complexity = compute_complexity(simplified)
            energy = estimate_energy(12, 1.0, 12e9, complexity)
            results.append({
                "prompt": simplified,
                "complexity": complexity,
                "energy": energy,
                "strategy": str(config)
            })
        except:
            pass
    return results
