from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from evaluate import load
from collections import Counter
import numpy as np
import torch
import pandas as pd
from scipy.stats import linregress

# Connect to Hugging Face
login(token="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")

# Instantiate Mistral 7B model
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Configure BitsAndBytes for 8-bit loading
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_dtype=torch.float16,
)

# Load model with offloading
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    offload_folder="offload",  
    torch_dtype=torch.float16
)

# Prompts
prompts = [
    "Describe a rainforest untouched by human hands and its unique ecosystem.",
    "Write a story about a magical sword that grants its wielder one wish.",
    "What is the meaning of happiness from the perspective of a timeless entity?",
    "Imagine you are a historian uncovering a lost diary from an ancient civilization.",
    "How can communities build resilience in the face of global challenges?"
]

# Sampling methods
strategies = [
    "Beam Search (b=4)",  
    "Pure Sampling",
    "Temperature (t=0.9)",
    "Top-k (k=640)",
    "Top-k with Temperature (k=40, t=0.7)",
    "Nucleus Sampling (p=0.95)"
]

# Decoding functions
def get_decoding_functions(inputs):
    return [
        lambda: model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=100,  
            num_beams=4,  
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        ),
        lambda: model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=100,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        ),
        lambda: model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=100,
            do_sample=True,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id
        ),
        lambda: model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=100,
            do_sample=True,
            top_k=640,
            pad_token_id=tokenizer.eos_token_id
        ),
        lambda: model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=100,
            do_sample=True,
            top_k=40,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        ),
        lambda: model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=100,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    ]

# Number of outputs per strategy per prompt
num_outputs = 10

# Generate outputs
all_outputs = {prompt: {strategy: [] for strategy in strategies} for prompt in prompts}
for i, prompt in enumerate(prompts, 1):
    print(f"Processing Prompt {i}/{len(prompts)}: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")  # Fix applied here
    decoding_functions = get_decoding_functions(inputs)
    for j, (strategy, decode_func) in enumerate(zip(strategies, decoding_functions), 1):
        print(f"  Generating outputs for Strategy {j}/{len(strategies)}: {strategy}")
        for k in range(num_outputs):
            torch.cuda.empty_cache()  # Clear cache to free memory
            output = decode_func()
            all_outputs[prompt][strategy].append(tokenizer.decode(output[0], skip_special_tokens=True))
            print(f"    Generated Output {k+1}/{num_outputs} for Strategy: {strategy}")

# Combine outputs
combined_outputs = {strategy: [] for strategy in strategies}
for prompt_outputs in all_outputs.values():
    for strategy, generations in prompt_outputs.items():
        combined_outputs[strategy].extend(generations)

# Perplexity
def perplexity(generations, model, tokenizer):
    combined_text = " ".join(generations)
    inputs = tokenizer(combined_text, return_tensors="pt", padding=True, truncation=True).to("cuda")  # Fix applied here
    inputs['labels'] = inputs['input_ids'].clone()
    with torch.no_grad():
        outputs = model(**inputs)
    loss = outputs.loss
    return torch.exp(loss).item() if loss else float('inf')

# Self-BLEU
def self_bleu(generations):
    bleu = load("bleu")
    scores = []
    for i, gen in enumerate(generations):
        references = generations[:i] + generations[i+1:]
        bleu.add(prediction=gen, references=references)
        scores.append(bleu.compute()["bleu"])
    return sum(scores) / len(scores)

# Repetition
def repetition(generations, n=3):
    combined_text = " ".join(generations)
    tokens = combined_text.split()
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    ngram_counts = Counter(ngrams)
    repeated = sum(count for count in ngram_counts.values() if count > 1)
    return (repeated / len(ngrams)) * 100 if len(ngrams) else 0

# Zipf coefficient
def zipf_coefficient(generations):
    combined_text = " ".join(generations)
    words = combined_text.split()
    counts = Counter(words)
    sorted_counts = sorted(counts.values(), reverse=True)
    ranks = range(1, len(sorted_counts) + 1)
    log_ranks = np.log(ranks)
    log_freqs = np.log(sorted_counts)
    slope, _, _, _, _ = linregress(log_ranks, log_freqs)
    return -slope

# Create dictionary for metrics
metrics = {
    "Strategy": [],
    "Perplexity": [],
    "Self-BLEU": [],
    "Repetition (%)": [],
    "Zipf Coefficient": []
}

# Add metrics to dictionary
for strategy, generations in combined_outputs.items():
    print(f"Computing metrics for Strategy: {strategy}")
    metrics["Strategy"].append(strategy)
    metrics["Perplexity"].append(perplexity(generations, model, tokenizer))
    metrics["Self-BLEU"].append(self_bleu(generations))
    metrics["Repetition (%)"].append(repetition(generations))
    metrics["Zipf Coefficient"].append(zipf_coefficient(generations))

# Convert dictionary to dataframe and print
metrics_df = pd.DataFrame(metrics)
print(metrics_df)

# Save metrics to a CSV file
metrics_df.to_csv("metrics.csv", index=False)
print("Metrics saved to 'metrics.csv'")