import numpy as np
import torch
from collections import Counter
from evaluate import load
from scipy.stats import linregress
from strategies import strategies

metrics = {
    "Strategy": [],
    "Perplexity": [],
    "Self-BLEU": [],
    "Repetition (%)": [],
    "Zipf Coefficient": []
}

def perplexity(prewritten_texts, model, tokenizer, strategy_func):
    perplexities = []
    for text in prewritten_texts:
        if text.strip(): 
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to("cuda")
            input_ids = inputs['input_ids']
            with torch.no_grad():
                logits = model(input_ids).logits  
            shift_logits = logits[:, :-1, :]  
            shift_labels = input_ids[:, 1:]  
            batch_perplexities = []
            for batch_idx in range(shift_logits.size(0)):  
                seq_logits = shift_logits[batch_idx] 
                seq_labels = shift_labels[batch_idx]  
                probabilities = strategy_func(seq_logits)  
                probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)        
                token_probs = probabilities[range(len(seq_labels)), seq_labels]       
                token_probs = token_probs + 1e-12          
                log_probs = torch.log(token_probs)          
                if torch.any(torch.isnan(log_probs)) or torch.any(torch.isinf(log_probs)):
                    continue      
                loss = -log_probs.mean()  
                perplexity = torch.exp(loss).item() 
                batch_perplexities.append(perplexity) 
            perplexities.extend(batch_perplexities)  
    return sum(perplexities) / len(perplexities) if perplexities else float('inf')

def self_bleu(all_outputs, target_strategy):
    strategy_self_bleu_scores = []  
    for prompt, prompt_outputs in all_outputs.items():
        if target_strategy not in prompt_outputs:
            continue  
        generations = prompt_outputs[target_strategy]
        scores = []  
        for i, gen in enumerate(generations):
            bleu = load("bleu")  
            references = generations[:i] + generations[i+1:]  
            if references:  
                bleu.add(prediction=gen, references=references)
                bleu_score = bleu.compute()["bleu"]
                scores.append(bleu_score)  
        if scores:  
            prompt_self_bleu = sum(scores) / len(scores)  
            strategy_self_bleu_scores.append(prompt_self_bleu)  
    if strategy_self_bleu_scores: 
        strategy_avg_self_bleu = sum(strategy_self_bleu_scores) / len(strategy_self_bleu_scores)
    else:
        strategy_avg_self_bleu = 0
    return strategy_avg_self_bleu

from collections import Counter

def repetition(generations, tokenizer, window_size=50, min_repeats=3, min_n=2):
    total_repeated_tokens = 0  
    total_tokens = 0  
    for generation in generations:
        tokens = tokenizer.tokenize(generation) 
        if len(tokens) == 0:
            continue
        window_tokens = tokens[-window_size:]
        for n in range(min_n, len(window_tokens) // min_repeats + 1):  
            ngrams = [tuple(window_tokens[i:i + n]) for i in range(len(window_tokens) - n + 1)]       
            i = 0
            while i < len(ngrams) - min_repeats + 1:
                current_ngram = ngrams[i]
                consecutive_repeats = 1
                for j in range(i + 1, len(ngrams)):
                    if ngrams[j] == current_ngram:
                        consecutive_repeats += 1
                    else:
                        break         
                if consecutive_repeats >= min_repeats:
                    repeated_tokens = consecutive_repeats * n  
                    total_repeated_tokens += repeated_tokens
                    i += consecutive_repeats * n  
                else:
                    i += 1
        total_tokens += len(tokens)  
    if total_tokens == 0:  
        return 0.0 
    repetition_percentage = (total_repeated_tokens / total_tokens) * 100
    return repetition_percentage

def zipf_coefficient(generations, tokenizer):
    combined_text = " ".join(generations)
    words = tokenizer.tokenize(combined_text)  
    counts = Counter(words)
    sorted_counts = sorted(counts.values(), reverse=True)
    if len(sorted_counts) < 2:
        return float('nan')  
    ranks = range(1, len(sorted_counts) + 1)
    log_ranks = np.log(ranks)
    log_freqs = np.log(sorted_counts)
    slope, _, _, _, _ = linregress(log_ranks, log_freqs)
    return -slope