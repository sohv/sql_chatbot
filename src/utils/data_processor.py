import json
from typing import List, Dict
from datasets import load_dataset
import os

def load_spider_dataset() -> Dict:
    dataset = load_dataset("spider")
    return dataset

def preprocess_spider_example(example: Dict) -> Dict:
    return {
        "question": example["question"],
        "query": example["query"],
        "db_id": example["db_id"],
        "schema": example.get("schema", ""),
        "tables": example.get("tables", []), 
        "columns": example.get("columns", [])  
    }

def create_training_examples(dataset: Dict, split: str = "train") -> List[Dict]:
    processed_dataset = dataset[split].map( # convert to list of dictionaries
        preprocess_spider_example,
        remove_columns=dataset[split].column_names
    )
    return [example for example in processed_dataset]

def save_examples(examples: List[Dict], output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(examples, f, indent=2)

def main():
    dataset = load_spider_dataset()
    
    train_examples = create_training_examples(dataset, "train")
    eval_examples = create_training_examples(dataset, "validation")
    
    save_examples(train_examples, "data/spider/train.json")
    save_examples(eval_examples, "data/spider/dev.json")
    
    print(f"Processed {len(train_examples)} training examples")
    print(f"Processed {len(eval_examples)} validation examples")

if __name__ == "__main__":
    main() 