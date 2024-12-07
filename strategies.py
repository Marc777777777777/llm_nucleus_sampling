from transformers import AutoModelForCausalLM

strategies = [
    "Beam Search (b=4)",  
    "Pure Sampling",
    "Temperature (t=0.9)",
    "Top-k (k=640)",
    "Top-k with Temperature (k=40, t=0.7)",
    "Nucleus Sampling (p=0.95)"
]

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