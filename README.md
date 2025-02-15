# CER
This is the offiicla repository for the CER: Confidence Enhanced Reasoning in LLMs paper.
Here is the instructions to use the code and reproduce any result from the paper. 

After you have clone the repo you can create your desired python virtual environment.
Then install the pacjage dependendcies by the follwoing comman 
```
pip install -r requirement.txt
```

After this step use the following command to download the spacy model.

```
python -m spacy download en_core_web_trf
```

Now crete the data directory in the root of the project. in the data directory you have to download the following files.
1.https://people.eecs.berkeley.edu/\~hendrycks/MATH.tar
2.http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json

You can use the follwoing commands.
```
mkdir data
cd data
wget https://people.eecs.berkeley.edu/\~hendrycks/MATH.tar
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json

tar xvf MATH.tar # to extrect the content of the zip file
```

Now you can create the datasets by running:
```
python src/custom_datasets/data_loader.py
```

yo create the mathematical datases and 
```
python src/custom_datasets/multihop_loader.py
```

for open domain generation question datasets.

All the configurations that are required in experiments are available in src/config.py file and you can modify it for your desired configuration. 

Also our code does support a .env file that contains the environment vcariables used in src/config.py you can set your desired environment variables in this file. 

After this step you can run the code simply by 
```
python main.py
```

Some important variables.
MODEL_NAME: name or the path to the desired huggingface model.
DATA_DIR: the absolute path of the data directory. e.g. /home/user/CER/data

RUN_NAME: specify the running mode "all" that means run all of configs of the multi_run_configs dictinory.
K: number of generated paths
aggregate: bool = True  # True: aggregate paths False: the best path
MULTIHOP: whether or not you try to run Trivia QA or HotPot QA datasets

N_SAMPLE: Number of samples to process
SEE: Seed for shuffling the dataset
BATCH_SIZE: the batch size of the inference.

STEP_DECOMPOSITION: use the incremental reasoning step prompt

DATASETS: a dictionary containing the dataseets and their coresspondig files. like """{
        "allenai": "allenai_math_qa_test_processed.parquet",
        "math": "src_datasets_math_dataset_test_processed.parquet",
        "gsm8k": "openai_gsm8k_test_processed.parquet",
        "hotpot": "hotpotqa_processed.parquet",
        "trivia": "triviaqa_processed.parquet",
    }"""

