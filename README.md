# CER: Confidence Enhanced Reasoning in LLMs

🎉 **We are pleased to announce that our paper has been accepted at the ACL 2025 Main Conference.**

Welcome to the official repository for the CER paper. This README provides step-by-step instructions to set up the environment, download necessary datasets, and reproduce the results presented in the paper.

---

## Table of Contents

- [CER: Confidence Enhanced Reasoning in LLMs](#cer-confidence-enhanced-reasoning-in-llms)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Configuration](#configuration)
  - [Running the Code](#running-the-code)
  - [Important Variables](#important-variables)

---

## Installation

1. **Clone the Repository:**  
   Start by cloning this repository to your local machine.

2. **Create a Python Virtual Environment:**  
   Set up a virtual environment of your choice to isolate package dependencies.

3. **Install Dependencies:**  
   Run the following command to install the required packages:
   ```bash
   pip install -r requirement.txt
   ```

4. **Download the spaCy Model:**  
   Download the `en_core_web_trf` model with:
   ```bash
   python -m spacy download en_core_web_trf
   ```

---

## Data Preparation

1. **Create Data Directory:**  
   In the root of the project, create a directory named `data`:
   ```bash
   mkdir data
   cd data
   ```

2. **Download Datasets:**  
   Download the following files into the `data` directory:
   - [MATH Dataset](https://people.eecs.berkeley.edu/~hendrycks/MATH.tar)
   - [HotPotQA Dataset](http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json)

3. **Extract and Prepare the Data:**  
   Use the commands below to download and extract the files:
   ```bash
   wget https://people.eecs.berkeley.edu/~hendrycks/MATH.tar
   wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json

   tar xvf MATH.tar  # Extract the MATH dataset
   ```

4. **Create the Datasets:**  
   Generate the datasets by running:
   ```bash
   python src/custom_datasets/data_loader.py
   ```
   for the mathematical datasets, and
   ```bash
   python src/custom_datasets/multihop_loader.py
   ```
   for the open-domain question generation datasets.

---

## Configuration

- **Experiment Settings:**  
  All necessary experiment configurations are defined in the `src/config.py` file. Modify this file to suit your requirements.

- **Environment Variables:**  
  The code supports a `.env` file. Set your desired environment variables in this file, which are then used in `src/config.py`.

---

## Running the Code

Once the environment is set up and data prepared, run the main program with:
```bash
python main.py
```

---

## Important Variables

- **MODEL_NAME:**  
  The name or path to the desired Hugging Face model.

- **DATA_DIR:**  
  The absolute path to the data directory (e.g., `/home/user/CER/data`).

- **RUN_NAME:**  
  Specifies the running mode. Use `"all"` to execute all configurations defined in the `multi_run_configs` dictionary.

- **K:**  
  The number of generated paths.

- **aggregate:**  
  A boolean flag indicating whether to aggregate paths (`True`) or select the best path (`False`).

- **MULTIHOP:**  
  Determines whether to run the Trivia QA or HotPot QA datasets.

- **N_SAMPLE:**  
  The number of samples to process.

- **SEED:**  
  The seed value used for shuffling the dataset.

- **BATCH_SIZE:**  
  The batch size used during inference.

- **STEP_DECOMPOSITION:**  
  Flag to use the incremental reasoning step prompt.

- **DATASETS:**  
  A dictionary mapping dataset names to their corresponding files. For example:
  ```python
  {
      "allenai": "allenai_math_qa_test_processed.parquet",
      "math": "src_datasets_math_dataset_test_processed.parquet",
      "gsm8k": "openai_gsm8k_test_processed.parquet",
      "hotpot": "hotpotqa_processed.parquet",
      "trivia": "triviaqa_processed.parquet",
  }
  ```
