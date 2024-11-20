import pandas as pd
import torch
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import numpy as np

dataset = load_dataset("Gryphe/Opus-WritingPromptsv", split='train')

model_name = "gpt2"
generator = pipeline('text-generation', model=model_name)

# Initialize BERT for BERTScore calculation
bert_model_name = "bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertForSequenceClassification.from_pretrained(bert_model_name)

def calculate_perplexity(generated_text):
    inputs = bert_tokenizer(generated_text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    log_likelihood = outputs.loss.item()
    perplexity = np.exp(log_likelihood)
    return perplexity

def calculate_bertscore(reference, generated):
    from bert_score import score
    P, R, F1 = score([generated], [reference], model_type=bert_model_name)
    return F1.mean().item()

def calculate_diversity(generated_texts):
    distinct_1 = len(set(generated_texts)) / len(generated_texts)  # Distinct-1
    ngrams = [tuple(generated_text.split()[i:i+2]) for generated_text in generated_texts for i in range(len(generated_text.split()) - 1)]
    distinct_2 = len(set(ngrams)) / len(ngrams) if ngrams else 0  # Distinct-2
    return distinct_1, distinct_2

def evaluate_model_on_prompts(prompts, num_return_sequences=1):
    results = []
    for prompt in prompts:
        generated_stories = generator(prompt, max_length=100, num_return_sequences=num_return_sequences)
        generated_texts = [story['generated_text'] for story in generated_stories]
        
        # Calculate metrics
        perplexities = [calculate_perplexity(text) for text in generated_texts]
        # bert_scores = [calculate_bertscore(prompt, text) for text in generated_texts]
        diversity_metrics = calculate_diversity(generated_texts)

        results.append({
            "prompt": prompt,
            "generated_stories": generated_texts,
            "perplexities": perplexities,
            # "bert_scores": bert_scores,
            "diversity_metrics": diversity_metrics,
        })
    return results

num_prompts_to_evaluate = 5
prompts_to_evaluate = dataset['prompt'][:num_prompts_to_evaluate].tolist()

evaluation_results = evaluate_model_on_prompts(prompts_to_evaluate)

# Display results
for result in evaluation_results:
    print(f"Prompt: {result['prompt']}")
    for i, story in enumerate(result['generated_stories']):
        print(f"Generated Story {i+1}: {story}")
        print(f"Perplexity: {result['perplexities'][i]}")
        # print(f"BERTScore: {result['bert_scores'][i]}")
    print(f"Diversity Metrics (Distinct-1, Distinct-2): {result['diversity_metrics']}\n")
