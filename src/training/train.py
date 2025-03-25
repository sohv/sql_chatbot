import os
import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
from datasets import load_dataset
from typing import Dict, Any
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        if 'hf_token' in config['model']:
            config['model']['hf_token'] = os.getenv('HUGGINGFACE_TOKEN')
        return config

def setup_model_and_tokenizer(config: Dict[str, Any]):
    login(token=config['model']['hf_token'])
    
    # 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['base_model'],
        trust_remote_code=config['model']['trust_remote_code'],
        use_auth_token=config['model']['use_auth_token']
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model'],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=config['model']['trust_remote_code'],
        use_auth_token=config['model']['use_auth_token']
    )

    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        **config['qlora'],
        inference_mode=False
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer

def format_prompt(example: Dict[str, Any]) -> str:
    """Format the prompt with schema information if available."""
    prompt = f"Question: {example['question']}\n"
    
    if example.get('schema'):
        prompt += f"Schema: {example['schema']}\n"
    if example.get('tables'):
        prompt += f"Tables: {', '.join(example['tables'])}\n"
    if example.get('columns'):
        prompt += f"Columns: {', '.join(example['columns'])}\n"
    
    prompt += f"SQL: {example['query']}"
    return prompt

def prepare_dataset(config: Dict[str, Any], tokenizer):
    dataset = load_dataset("spider")
    
    def preprocess_function(examples):
        prompts = [format_prompt(example) for example in examples]
        
        return tokenizer(
            prompts,
            truncation=True,
            max_length=config['dataset']['max_length'],
            padding=config['dataset']['padding']
        )

    train_dataset = dataset['train'].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    eval_dataset = dataset['validation'].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['validation'].column_names
    )

    return train_dataset, eval_dataset

def main():
    config = load_config('configs/training_config.yaml')

    model, tokenizer = setup_model_and_tokenizer(config)

    train_dataset, eval_dataset = prepare_dataset(config, tokenizer)

    training_args = TrainingArguments(
        **config['training'],
        report_to="wandb",
        remove_unused_columns=False,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()

    trainer.save_model()

if __name__ == "__main__":
    main()