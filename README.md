# SQL Chatbot with Llama 3.3 and QLoRA

This project implements a SQL chatbot using Llama 3.3 70B Instruct model fine-tuned with QLoRA (Quantized Low-Rank Adaptation) for optimized SQL query generation.

## Project Structure
```
sql_chatbot/
├── data/                   # Training and evaluation datasets
├── src/                    # Source code
│   ├── training/          # Training scripts
│   ├── inference/         # Inference scripts
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── models/                # Saved model checkpoints
└── notebooks/             # Jupyter notebooks for analysis
```

## Prerequisites

1. Request access to Llama 3.3:
   - Go to https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
   - Click "Access repository"
   - Fill out Meta AI's form to request access
   - Accept the license agreement
   - Wait for approval email from Meta AI

2. Set up HuggingFace API token:
   - Create a HuggingFace account at https://huggingface.co/
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Give your token a name (e.g., "SQL Chatbot")
   - Select "read" role
   - Copy the generated token

## Setup

1. Clone the repository:
```bash
git clone https://github.com/sohv/sql_chatbot.git
cd sql_chatbot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env


```

The `.env` file should look like this:
```
HUGGINGFACE_TOKEN=your_actual_token_here
```

## Training

1. Process the dataset:
```bash
python src/utils/data_processor.py
```

2. Fine-tune the model using QLoRA:
```bash
python src/training/train.py
```

## Inference

To run the Streamlit interface:
```bash
streamlit run src/inference/app.py
```

## Dataset

We use the [Spider](https://yale-lily.github.io/spider) dataset for fine-tuning, which contains complex SQL queries and their corresponding natural language questions.

## Model Architecture

- Base Model: Llama 3.3 70B Instruct
- Fine-tuning Method: QLoRA
- Quantization: 4-bit
- LoRA Parameters:
  - r: 8
  - alpha: 16
  - dropout: 0.05

## Troubleshooting

1. If you get a 401 Unauthorized error:
   - Ensure you have requested and received access to Llama 3.3
   - Verify your HuggingFace token is correct in the .env file
   - Make sure you've accepted the model's license agreement

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or create a pull request.

## Acknowledgments

- Meta AI for Llama 3.3
- HuggingFace for the Transformers library
- Yale LILY Lab for the Spider dataset
- The open-source community for valuable resources and tools
