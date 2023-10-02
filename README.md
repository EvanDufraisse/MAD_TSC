# MAD-TSC: A Multilingual Aligned News Dataset for Target-dependent Sentiment Classification

Link to ACL 2023 related paper: [paper](https://aclanthology.org/2023.acl-long.461)

MAD-TSC is a multilingual aligned dataset for target-dependent sentiment classification (TSC) on the news domain.
Target-dependent classification (TSC) is closely related to the task of Aspect-Based Sentiment Classification (ABSC), a classification step in the overall study of Aspect-Based Sentiment Analysis (ABSA).

The aim of TSC is to determine the sentiment expressed towards a target in a given context:

~~~
"I think <John> hasn't done a great work compared to Sam." -> Sentiment expressed towards <John> is negative
~~~

- The dataset contains 5 110 entries  1 839 negative, 2 011 neutral and 1 260 positive.

The dataset was built using a European initiative of professionally translated news articles [VoxEurop](https://voxeurop.eu).

The aligned nature of the dataset allows for multilingual experiments, as is shown in the paper. In the context of those experiments, parts of the dataset have also been translated to evaluate performances of automatic translation algorithms (M2M100 and DeepL).

## Dataset

The dataset is available in the `data` folder. The `data` folder contains the following files:

```bash
- m2m12b # Translated using m2m-100 12B model
    - from_<src lang>_to_<tgt lang>
        - train.jsonl
        - test.jsonl
        - validation.jsonl
- deepl # Translated using deepl API
    - from_<src lang>_to_<tgt lang>
        - train.jsonl
        - test.jsonl
        - validation.jsonl
- original # Original dataset translated by professionals and aligned automatically
    - <lang>
        - train.jsonl
        - test.jsonl
        - validation.jsonl
```

## Reproducibility

For reproducibility purposes, we provide code to reproduce the experiments.
Code was made on a cluster, with file transfers between scratch and home directories, the code makes use of rsync, which is not available on Windows to my knowledge. The code is not guaranteed to work on Windows.
### Installation

Activate your virtual environment and install the requirements using pip:

```bash
pip install -e ./setup.py # (or pip install ./setup.py if you don't want to edit the code)
```

### Launch Training and Testing

Example bash script is given in folder scripts

The command to launch finetuning is the following:

You must complete the following <> fields with your own values:


```bash

#!/bin/bash


eval "$(conda shell.bash hook)"

conda activate tscbench

export UUID=$RANDOM
export TOKENIZERS_PARALLELISM=false
export DATA_SCRATCH=<path to put temporary input files>
export EXPERIMENT_DIR=<path to the output of the experiments>
export EXPERIMENT_SCRATCH=<path with the most I/O operation>
rsync -avh --progress <path to your dataset root folder> "/scratch/data/"$UUID"/"


tscbench finetune tsc \
-n <name of the experiment> \
-m <path to model> \
-t <path to tokenizer> \
--tsc-model-config <path to tsc model configuration> \
--dataset-config <path to dataset configuration> \
--gpu-pl-config <path to gpu config> \
--sub-path-final-folder <subpath where the output model and statistiqcs> \
--optimizer-config <path to optimizer configuration> \
--keep-best-models # Keep best model for a given run, --keep-all-models to keep all models checkpoints
```

Also check the config readme to configure for other models and datasets

The results will be stored in the folder of the experiment under $EXPERIMENT_DIR

Independent inference script is not provided yet...
