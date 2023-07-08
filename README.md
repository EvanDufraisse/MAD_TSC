# MAD-TSC: A Multilingual Aligned News Dataset for Target-dependent Sentiment Classification

## DATASET

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