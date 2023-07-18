# Configuration files

Several configuration files are necessary as input for the model.
Below we describe the different configuration files and their content.

Variables refered as <NAME_OF_VARIABLE> are the one supplied as part of the command line arguments.

## Dataset configuration

~~~ json
{
            "name_dataset":"MAD_TSC_en", # Custom name for the dataset
            "format":"newsmtsc", # Default format for the dataset, other format were not used as part of paper experiments
            "folder_dataset":"MAD_TSC_en", # Folder where the dataset is located relative to $SCRATCH_DATA
            "filenames":{"train":"train.jsonl", "test":"test.jsonl", "validation":"validation.jsonl"}, # Filenames for the different splits, not all splits may be supplied
            "splitting_strategy":{"train":[1,0,0], "test":[0,1,0], "validation":[0,0,1]} # Determine the proportion of train / test / validation data to use for each split, shouldn't be modified if all splits are supplied as filenames
        }
~~~

## Lightning configuration

~~~ json
{
    "pytorch_lightning_flags":{
        "val_check_interval": 0.33, # Run validation every 33% of an epoch
        "log_every_n_steps": 50, # Log every 50 steps
        "benchmark": true, # For performance, if the input size is constant
        "devices":1, # Number of GPUs to use
        "accelerator":"gpu" # Use GPU
    },
    "pytorch_lightning_params": {
        "log_dir": "logs/", # Folder where to store the logs as a children to $EXPERIMENT_DIR/<SUB_PATH_FINAL_FOLDER>/<NAME_EXPERIMENT>
        "path_checkpoints": "checkpoints/", # Folder where to store the checkpoints as a children to $EXPERIMENT_DIR/<SUB_PATH_FINAL_FOLDER>/<NAME_EXPERIMENT>
        "checkpoints_topk": 1, # Number of checkpoints to keep
        "lr_monitor": true # Monitor the learning rate
    }
}
~~~

## Models configuration

Configurations of models used in the paper are supplied.

- SPC model : "spc_model_default.json"
- TD model : "td_model_default.json"
- Prompt model: "prompt_model_default_[lang].json" modify according to your language


## Optimizers configuration

Define the training hyperparameters and the optimizer to use.
We only comment the parts that the user may want to modify.

~~~ json
{
    "optimizer":{
        "scheduler": "warmup_linear", # Type of scheduler to use
        "hyperparameters": {
            "batch_size": [32], # Effective batch size, no vram consideration
            "lr": [2e-5], # Learning rate
            "epochs": [40], # Number of epochs to train
            "seeds": [42, 302, 668, 745, 343] # Seeds to use for the different runs
        },
        "checkpoint_metrics":{"validation_loss":"min", "validation_f1score":"max"},
        "optuna_metrics": {"test_loss":"min", "test_f1score":"max"},
        "adam_epsilon": 1e-6,
        "weight_decay": 0.01,
        "proportion_warmup": 0.06,
        "precision": 16, # 16 for float16, 32 for float32
        "max_batch_size_per_gpu": 16, # Maximum batch size per GPU, vram consideration
        "num_workers": 2, # Number of workers for the dataloader
        "direction": "min", # Optuna setting
    },
    "freeze_layers": [],
    "freeze_representation_layer":false,
    "partial_gradient_wordsembeddings":false
}
~~~

