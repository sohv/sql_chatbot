import os
import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from datasets import load_dataset
from typing import Dict, Any, List, Tuple
from huggingface_hub import login
from dotenv import load_dotenv
import sqlite3
import pandas as pd
from tqdm import tqdm
import json

load_dotenv()

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        if 'hf_token' in config['model']:
            config['model']['hf_token'] = os.getenv('HUGGINGFACE_TOKEN')
        return config

def setup_model_and_tokenizer(config: Dict[str, Any]):
    login(token=config['model']['hf_token'])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        quantization_config = None
        print("Warning: Running on CPU. Evaluation will be slower.")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['base_model'],
        trust_remote_code=config['model']['trust_remote_code'],
        use_auth_token=config['model']['use_auth_token']
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model'],
        quantization_config=quantization_config,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=config['model']['trust_remote_code'],
        use_auth_token=config['model']['use_auth_token']
    )

    # loading the LoRA weights
    model = PeftModel.from_pretrained(model, config['training']['output_dir'])
    model.eval()

    return model, tokenizer

def format_prompt(example: Dict[str, Any]) -> str:
    prompt = f"Question: {example['question']}\n"
    
    if example.get('schema'):
        prompt += f"Schema: {example['schema']}\n"
    if example.get('tables'):
        prompt += f"Tables: {', '.join(example['tables'])}\n"
    if example.get('columns'):
        prompt += f"Columns: {', '.join(example['columns'])}\n"
    
    prompt += "SQL:"
    return prompt

def generate_sql(model, tokenizer, prompt: str, max_length: int = 2048) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql_query = generated_text.split("SQL:")[-1].strip()
    
    return sql_query

def normalize_sql(sql: str) -> str:
    sql = ' '.join(line for line in sql.split('\n') if not line.strip().startswith('--'))
    sql = ' '.join(sql.split())
    sql = sql.lower()
    return sql

def evaluate_sql_execution(predicted_sql: str, expected_sql: str, db_path: str) -> Tuple[bool, str]:
    try:
        conn = sqlite3.connect(db_path)
        
        pred_df = pd.read_sql_query(predicted_sql, conn)
        exp_df = pd.read_sql_query(expected_sql, conn)
        
        is_correct = pred_df.equals(exp_df)
        error_msg = "" if is_correct else "Results don't match"
        
        conn.close()
        return is_correct, error_msg
        
    except Exception as e:
        return False, str(e)

def evaluate_model(model, tokenizer, dataset, config: Dict[str, Any]) -> Dict[str, Any]:
    results = {
        'total_samples': len(dataset),
        'exact_match': 0,
        'execution_correct': 0,
        'errors': []
    }
    
    for example in tqdm(dataset, desc="Evaluating"):
        # generate SQL query
        prompt = format_prompt(example)
        predicted_sql = generate_sql(model, tokenizer, prompt)
        
        pred_sql_norm = normalize_sql(predicted_sql)
        exp_sql_norm = normalize_sql(example['query'])
        
        is_exact_match = pred_sql_norm == exp_sql_norm
        if is_exact_match:
            results['exact_match'] += 1
        
        is_exec_correct, error_msg = evaluate_sql_execution(
            predicted_sql, 
            example['query'],
            f"data/spider/database/{example['db_id']}/{example['db_id']}.sqlite"
        )
        
        if is_exec_correct:
            results['execution_correct'] += 1
        else:
            results['errors'].append({
                'question': example['question'],
                'predicted_sql': predicted_sql,
                'expected_sql': example['query'],
                'error': error_msg
            })
    
    results['exact_match_accuracy'] = results['exact_match'] / results['total_samples']
    results['execution_accuracy'] = results['execution_correct'] / results['total_samples']
    
    return results

def save_results(results: Dict[str, Any], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    config = load_config('configs/training_config.yaml')
    
    model, tokenizer = setup_model_and_tokenizer(config)
    
    dataset = load_dataset("spider", split="validation")
    
    results = evaluate_model(model, tokenizer, dataset, config)
    
    print("\nEvaluation Results:")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Exact Match Accuracy: {results['exact_match_accuracy']:.2%}")
    print(f"Execution Accuracy: {results['execution_accuracy']:.2%}")
    print(f"Number of Errors: {len(results['errors'])}")
    
    output_path = os.path.join(config['training']['output_dir'], 'eval_results.json')
    save_results(results, output_path)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main() 