import os
import json
import datasets
from dotenv import load_dotenv

'''
Dataset url https://people.eecs.berkeley.edu/\~hendrycks/MATH.tar
'''

load_dotenv(override=True)  # Reads .env file and loads environment variables


class MathDatasetConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(MathDatasetConfig, self).__init__(**kwargs)


class MathDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        MathDatasetConfig(name="default", version=datasets.Version("1.0.0"), description="MATH Dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="Mathematical problem-solving dataset with problems and solutions.",
            features=datasets.Features(
                {
                    "problem": datasets.Value("string"),
                    "solution": datasets.Value("string"),
                    "level": datasets.Value("string"),
                    "type": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        data_dir = os.getenv("DATA_DIR", "data") + "/MATH"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"directory": os.path.join(data_dir, "train")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"directory": os.path.join(data_dir, "test")}
            ),
        ]

    def _generate_examples(self, directory):
        idx = 0
        for category in os.listdir(directory):
            category_dir = os.path.join(directory, category)
            if not os.path.isdir(category_dir):
                continue
            for filename in sorted(os.listdir(category_dir)):
                filepath = os.path.join(category_dir, filename)
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                    yield idx, {
                        "problem": data.get("problem", ""),
                        "solution": data.get("solution", ""),
                        "level": data.get("level", ""),
                        "type": data.get("type", category),
                    }
                    idx += 1
