from transformers import AutoModelForCausalLM
import torch
import torch.nn.functional as F

# List of strategies
strategies = [
    "Beam Search (b=4)",  
    "Pure Sampling",
    "Temperature (t=0.9)",
    "Top-k (k=640)",
    "Top-k with Temperature (k=40, t=0.7)",
    "Nucleus Sampling (p=0.95)"
]

# Function that returns decoding function for all strategies
def get_decoding_functions(inputs, model, tokenizer):
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

import torch
import torch.nn.functional as F

# Beam search sampling function
def beam_search_strategy(logits, num_beams=4):
    top_k_values, top_k_indices = torch.topk(logits, num_beams, dim=-1)
    top_k_probs = F.softmax(top_k_values, dim=-1)
    prob_mask = torch.zeros_like(logits).scatter_(-1, top_k_indices, top_k_probs)
    prob_mask = prob_mask / prob_mask.sum(dim=-1, keepdim=True)  # Normalise
    prob_mask = prob_mask + 1e-12  # Avoid log(0)
    return prob_mask

# Pure sampling function
def pure_sampling_strategy(logits):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    probabilities = probabilities + 1e-12  # Avoid log(0)
    return probabilities

# Temperature sampling function
def temperature_strategy(logits, temperature=0.9):
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=-1)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    probabilities = probabilities + 1e-12  # Avoid log(0)
    return probabilities

# Top-k sampling function
def top_k_strategy(logits, k=640):
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    top_k_probs = F.softmax(top_k_values, dim=-1)
    prob_mask = torch.zeros_like(logits).scatter_(-1, top_k_indices, top_k_probs)
    prob_mask = prob_mask / prob_mask.sum(dim=-1, keepdim=True)  # Normalise
    prob_mask = prob_mask + 1e-12  # Avoid log(0)
    return prob_mask

# Top-k with temperature sampling function
def top_k_with_temperature_strategy(logits, k=40, temperature=0.7):
    scaled_logits = logits / temperature
    top_k_values, top_k_indices = torch.topk(scaled_logits, k, dim=-1)
    top_k_probs = F.softmax(top_k_values, dim=-1)
    prob_mask = torch.zeros_like(logits).scatter_(-1, top_k_indices, top_k_probs)
    prob_mask = prob_mask / prob_mask.sum(dim=-1, keepdim=True)  # Normalise
    prob_mask = prob_mask + 1e-12  # Avoid log(0)
    return prob_mask

# Nucleus sampling function
def nucleus_strategy(logits, p=0.95):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    mask = cumulative_probs > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits[mask] = -float('Inf')
    probabilities = F.softmax(sorted_logits, dim=-1)
    prob_mask = torch.zeros_like(logits).scatter_(-1, sorted_indices, probabilities)
    prob_mask = prob_mask / prob_mask.sum(dim=-1, keepdim=True)  # Normalise
    prob_mask = prob_mask + 1e-12  # Avoid log(0)
    return prob_mask

# Dictionnary of sampling functions
strategy_funcs = {
    "Beam Search (b=4)": lambda logits: beam_search_strategy(logits, num_beams=4),
    "Pure Sampling": lambda logits: pure_sampling_strategy(logits),
    "Temperature (t=0.9)": lambda logits: temperature_strategy(logits, temperature=0.9),
    "Top-k (k=640)": lambda logits: top_k_strategy(logits, k=640),
    "Top-k with Temperature (k=40, t=0.7)": lambda logits: top_k_with_temperature_strategy(logits, k=40, temperature=0.7),
    "Nucleus Sampling (p=0.95)": lambda logits: nucleus_strategy(logits, p=0.95)
}