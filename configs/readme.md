# Configuration files

Several configuration files are necessary as input for the model.
Below we describe the different configuration files and their content.


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

## Models configuration

## Optimizers configuration


