# MAD-TSC: A Multilingual Aligned News Dataset for Target-dependent Sentiment Classification

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

You must beforehand define three environment variables and create the corresponding folders:


```bash
export DATA_SCRATCH=<path to put temporary input files>
export EXPERIMENT_DIR=<path to the output of the experiments>
export EXPERIMENT_SCRATCH=<path with the most I/O operation>

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