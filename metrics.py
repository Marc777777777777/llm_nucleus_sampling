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

def perplexity(prewritten_texts, model, tokenizer):
    perplexities = []
    for text in prewritten_texts:
        if text.strip():  
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to("cuda")
            inputs['labels'] = inputs['input_ids'].clone()            
            with torch.no_grad():
                outputs = model(**inputs)         
            loss = outputs.loss
            if loss: 
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)
    return sum(perplexities) / len(perplexities) if perplexities else float('inf')

def self_bleu(all_outputs):
    strategy_self_bleu_scores = {strategy: [] for strategy in strategies}
    for prompt, prompt_outputs in all_outputs.items():
        for strategy, generations in prompt_outputs.items():
            scores = []
            for i, gen in enumerate(generations):
                bleu = load("bleu")
                references = generations[:i] + generations[i+1:]  
                bleu.add(prediction=gen, references=references)
                scores.append(bleu.compute()["bleu"])
            prompt_self_bleu = sum(scores) / len(scores)
            strategy_self_bleu_scores[strategy].append(prompt_self_bleu)  
    strategy_avg_self_bleu = {strategy: sum(scores) / len(scores) for strategy, scores in strategy_self_bleu_scores.items()} 
    return strategy_avg_self_bleu

def repetition(generations, tokenizer, min_phrase_length=2, min_repeats=3, window_size=200):
    total_tokens = 0  
    repeated_tokens = 0  
    for text in generations:
        tokens = tokenizer.tokenize(text)
        if len(tokens) > window_size:
            tokens = tokens[-window_size:]
        for n in range(min_phrase_length, len(tokens) + 1):
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            i = 0
            while i < len(ngrams) - 1:
                current_ngram = ngrams[i]
                consecutive_repeats = 1
                while i + consecutive_repeats < len(ngrams) and ngrams[i + consecutive_repeats] == current_ngram:
                    consecutive_repeats += 1
                if consecutive_repeats >= min_repeats:
                    repeated_tokens += n * consecutive_repeats
                i += consecutive_repeats  
        total_tokens += len(tokens)
    if total_tokens == 0:
        return 0.0
    return (repeated_tokens / total_tokens) * 100

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