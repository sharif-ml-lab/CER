# REPLUG Project Setup

## Project Structure

```bash
REPLUG_Project/
│
├── retriever/
│   ├── retriever.py          # Code for dense retrieval model
│   ├── train_retriever.py    # Training code for the dense retriever with LM-Supervised Retrieval (LSR)
│   └── faiss_index.py        # Code to build and update FAISS index
│
├── replug/
│   ├── inference.py          # Inference pipeline for REPLUG
│   ├── lm_api.py             # API interaction to query the frozen black-box LM (e.g., GPT-3)
│   └── input_reformulation.py # Code for input reformulation and ensemble method
│
├── utils/
│   ├── data_loader.py        # Utilities to load data (e.g., document corpus, queries)
│   ├── similarity.py         # Helper functions for cosine similarity and document embedding
│   └── evaluation.py         # Functions to evaluate performance on different tasks
│
└── main.py                   # Main script for training and inference operations
```

## Setup Instructions

1. **Install Python Dependencies**

   Make sure to have Python 3.7+ installed. You can install all the necessary dependencies using the provided `requirements.txt` file. Run the following command:

   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up OpenAI API Key**

   To use the OpenAI GPT-3 model, make sure you have an API key from OpenAI. Set the API key as an environment variable using the following command:

   ```bash
   export OPENAI_API_KEY='your_openai_api_key_here'
   ```

3. **Directory Structure**

   Create the project directory and the respective folder structure as described above.

4. **Running the Retriever Training**

   To train the retriever model, use the following command:

   ```bash
   python retriever/train_retriever.py
   ```

5. **Running Inference**

   To run the REPLUG inference, use the following command:

   ```bash
   python replug/inference.py
   ```

## Notes

- The retriever model is trained to adaptively learn to retrieve documents based on LM scoring.
- The inference script makes use of the retriever to get relevant documents and then passes them to the black-box LM for better predictions.
- Modify the paths and API keys accordingly in the scripts to make sure they match your local environment.
