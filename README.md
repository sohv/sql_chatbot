## SQL Query Generator Chatbot ##

This repository contains the code and resources for building a chatbot that generates SQL queries based on user prompts. We fine-tuned the GPT-2 model on a dataset sourced from the Hugging Face library and trained it using TensorFlow. The fine-tuned model is saved in a designated folder within this repository.

## Requirements: ##

* Python 3.x
* TensorFlow
* Hugging Face Transformers
* Other dependencies as specified in requirements.txt

## Installation: ##

1. Clone this repository to your local machine:

git clone https://github.com/sohv/sql_chatbot.git

2. Navigate to the cloned directory:
   
cd sql_chatbot

3. Install the required dependencies:

pip install -r requirements.txt

## Usage: ##
Ensure that the fine-tuned GPT-2 model is saved in the designated folder (models/ by default).

Run the chatbot script:<br>
python chatbot.py <br>
The chatbot will prompt you to enter your query requirements. Based on your input, it will generate SQL queries.

## Training: ##
If you want to fine-tune the model further or train it from scratch:

*Prepare your dataset in the required format.
*Use the provided scripts or adapt them for your dataset.
*Fine-tune the GPT-2 model using TensorFlow.

## Model ##
The fine-tuned GPT-2 model is saved in the models/ folder. This model is used by the chatbot to generate SQL queries based on user input.

## Contributing ##
Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or create a pull request.


Acknowledgments
This project utilizes the Hugging Face Transformers library.
Inspiration for this project came from the need for an intelligent SQL query generator.
Thanks to the open-source community for providing valuable resources and tools.
