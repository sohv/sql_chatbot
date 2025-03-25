import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_model(config):
    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model'],
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=config['model']['trust_remote_code']
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['base_model'],
        trust_remote_code=config['model']['trust_remote_code']
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, config['training']['output_dir'])
    
    return model, tokenizer

def generate_sql(model, tokenizer, question, max_length=2048):
    prompt = f"Question: {question}\nSQL:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    st.title("SQL Query Generator")
    st.write("Ask a question and get the corresponding SQL query")
    
    # Load configuration
    config = load_config('configs/training_config.yaml')
    
    # Setup model
    model, tokenizer = setup_model(config)
    
    # User input
    question = st.text_input("Enter your question:")
    
    if st.button("Generate SQL"):
        if question:
            with st.spinner("Generating SQL query..."):
                sql = generate_sql(model, tokenizer, question)
                st.code(sql, language="sql")
        else:
            st.warning("Please enter a question")

if __name__ == "__main__":
    main() 