#
# Created on Thu Jun 30 2022
#
# Copyright (c) 2022 CEA - LASTI
# Contact: Evan Dufraisse,  evan.dufraisse@cea.fr. All rights reserved.
#

from abc import ABC, abstractmethod
from multiprocessing.sharedctypes import Value
import torch
import jsonlines
import os
import random
import logging
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import AutoTokenizer
from tscbench.utils.widgets import CustomTqdm

# region [yellow]

# Ensemble of registered datasets
REGISTERED_DATASETS = {
    "newsmtscmt": {
        "format": "newsmtsc",
        "folder_dataset": "newsmtsc-train-and-test-mt",
        "filenames": {"train": "train.jsonl", "test": "devtest_mtsc_only.jsonl"},
        "splitting_strategy": {"train": [1, 0, 0], "test": [0, 0.7, 0.3]},
    },
    "newsmtscrw": {
        "format": "newsmtsc",
        "folder_dataset": "newsmtsc-train-and-test-rw",
        "filenames": {
            "train": "train.jsonl",
            "test": "devtest_mtsc_and_single_primaries.jsonl",
        },
        "splitting_strategy": {"train": [1, 0, 0], "test": [0, 0.7, 0.3]},
    },
    "laptop": {
        "format": "newsmtsc",
        "folder_dataset": "regenerated_semeval14laptops",
        "filenames": {
            "train": "Laptops_Train.xml.seg.jsonl",
            "test": "Laptops_Test_Gold.xml.seg.jsonl",
        },
        "splitting_strategy": {"train": [0.8, 0, 0.2], "test": [0, 1, 0]},
    },
    "restaurants": {
        "format": "newsmtsc",
        "folder_dataset": "regenerated_semeval14restaurants",
        "filenames": {
            "train": "restaurants_train.jsonl",
            "test": "restaurants_test.jsonl",
        },
        "splitting_strategy": {"train": [0.8, 0, 0.2], "test": [0, 1, 0]},
    },
    "twitter": {
        "format": "newsmtsc",
        "folder_dataset": "acl14twitter",
        "filenames": {"train": "train.raw.jsonl", "test": "test.raw.jsonl"},
        "splitting_strategy": {"train": [0.8, 0, 0.2], "test": [0, 1, 0]},
    },
}

# endregion


# region [tested]
class RawDataLoader(ABC):
    def __init__(self, root_folder, name_dataset):
        self.root_folder = root_folder
        self.name_dataset = name_dataset

    @abstractmethod
    def load_data(self, filenames):
        pass


class RawJsonlLoader(RawDataLoader):
    """
    Load jsonlines dataset
    """

    def __init__(self, root_folder, name_dataset):
        super().__init__(root_folder, name_dataset)

    def load_data(self, filenames: dict) -> "list[dict]":
        data = {}
        for name, filename in filenames.items():
            data[name] = []
            with jsonlines.open(
                os.path.join(
                    os.path.join(self.root_folder, self.name_dataset), filename
                )
            ) as reader:
                for obj in reader:
                    data[name].append(obj)

        return data


# endregion


class DataSplitter(object):
    """
    Takes data as input and returns a list of lists of data, where each list of data is a split of the original data.
    """

    def __init__(self, data, splitting_strategy, seed=42):
        self.data = data
        self.splitting_strategy = splitting_strategy
        self.seed = seed

    def _subsplit(self, subdata, i1, i2, seed):
        """splits a subdata in two subdata according to the proportions i1 and i2

        Args:
            subdata (list): list of data to split
            i1 (float): proportion of original data to keep in the first subdata
            i2 (float): proportion of original data to keep in the second subdata
            seed (int): seed for the random generator

        Returns:
            tuple[list]: splitted data
        """
        if i2 == 0:
            return subdata, []
        i1 = i1 / i2
        if i1 == 1:
            return subdata, []
        elif i1 == 0:
            return [], subdata
        t_2, t_1 = train_test_split(subdata, test_size=i1, random_state=seed)
        return t_1, t_2

    def split_data(self, data, strategy, seed) -> "list[dict]":
        """Splits data according to splitting_strategy.

        Args:
            data (list): list of data to split
            strategy (list[float]): list of proportions to split the data into.
            seed (int): seed  for the random generator

        Returns:
            list[list]: list of lists of data, where each list of data is a split of the original data.
        """
        random.seed(seed)
        seeds = [random.randint(0, 10_000) for _ in range(len(strategy))]
        cumsum = np.cumsum(strategy)
        obtained_splits = []
        last_split = data
        for k in range(cumsum.shape[0] - 2, -1, -1):
            i1, i2 = tuple(cumsum[k : k + 2])
            t_1, t_2 = self._subsplit(last_split, i1, i2, seeds[k])
            last_split = t_1
            obtained_splits.append(t_2)
        obtained_splits.append(last_split)
        return obtained_splits[::-1]

    # region [core]
    def get_new_data_splits(self, data, splitting_strategy, seed) -> "list[dict]":
        """Returns a list of lists of data, where each list of data is a split of the original data.

        Args:
            data (dict[str,list]): mapping from dataset name to list of data to split
            splitting_strategy (dict[str,list[float]]): mapping from dataset name to list of proportions to split the data into.
            seed (int): seed for the random generator

        Returns:
            list[list]: list of lists of data, where each list of data is a split of the original data in supplied proportions.
        """
        random.seed(seed)
        new_splits = [[]] * len(splitting_strategy[list(splitting_strategy.keys())[0]])
        for name, data_init_split in data.items():
            temp = self.split_data(data_init_split, splitting_strategy[name], seed)
            for k in range(len(temp)):
                new_splits[k] = new_splits[k] + temp[k]
        for k in range(len(new_splits)):
            for _ in range(5):
                random.shuffle(new_splits[k])
        return new_splits


#  endregion


# region [tested]
class AbsaDatasetLoader(object):
    def __init__(self, root_folder, name_dataset, seed=42):
        self.root_folder = root_folder
        self.name_dataset = name_dataset
        if not (self.name_dataset in REGISTERED_DATASETS):
            raise ValueError("Dataset {} is not supported".format(self.name_dataset))
        self.seed = seed

    def load_dataset(self):
        """Loads the dataset."""
        dataset = REGISTERED_DATASETS[self.name_dataset]
        loader = RawJsonlLoader(self.root_folder, dataset["folder_dataset"])
        data = loader.load_data(dataset["filenames"])
        splitter = DataSplitter(data, dataset["splitting_strategy"], self.seed)
        final_data = splitter.get_new_data_splits(
            data, dataset["splitting_strategy"], self.seed
        )
        return final_data


# endregion


# def return_tokens_chars(out, chars):
#     return (out.char_to_token(chars[0]), out.char_to_token(chars[1]-1)+1)
class PromptEncoder(object):
    def __init__(self):
        pass

    def encode_entry(
        self,
        mention,
        text,
        model_prompt_template,
        tokenizer,
        return_tensors=None,
        padding="do_not_pad",
        truncation=False,
        add_special_tokens=False,
        encode_plus=False,
        return_mention_tokens=False,
        multi_prompt_fusion="sum",
    ):
        if not (text.endswith(".")):
            text += "."
        if type(model_prompt_template) == list:
            all_encoded = []
            for prompt_template in model_prompt_template:
                prompt_template = model_prompt_template
                prompt_template = prompt_template.replace("<entity>", mention)
                prompt_template = prompt_template.replace("<aspect>", mention)
                prompt_template = prompt_template.replace(
                    "<mask>", tokenizer.mask_token
                )
                if len(prompt_template) > 0:
                    full_text = text + " " + prompt_template
                else:
                    full_text = text
                if encode_plus or return_mention_tokens:
                    all_encoded.append(
                        tokenizer.encode_plus(
                            full_text,
                            add_special_tokens=add_special_tokens,
                            return_tensors=return_tensors,
                            padding=padding,
                            truncation=truncation,
                        )
                    )
                else:
                    all_encoded.append(
                        tokenizer.encode(
                            full_text,
                            add_special_tokens=add_special_tokens,
                            return_tensors=return_tensors,
                            padding=padding,
                            truncation=truncation,
                        )
                    )
            return all_encoded
        else:
            prompt_template = model_prompt_template
            prompt_template = prompt_template.replace("<entity>", mention)
            prompt_template = prompt_template.replace("<mask>", tokenizer.mask_token)
            if len(prompt_template) > 0:
                full_text = text + " " + prompt_template
            else:
                full_text = text
            # if return_mention_tokens:
            #     # if mentions_pos is None:
            #     #     raise ValueError("mentions_pos is None")
            #     return tokenizer.encode_plus(full_text, add_special_tokens=add_special_tokens, return_tensors=None)
            # mention_tokens = []
            # for (start, end) in mentions_pos:
            #     token_start, token_end = return_tokens_chars(out, (start, end))
            #     mention_tokens.append((token_start, token_end))
            # return out, mention_tokens
            if encode_plus or return_mention_tokens:
                return tokenizer.encode_plus(
                    full_text,
                    add_special_tokens=add_special_tokens,
                    return_tensors=return_tensors,
                    padding=padding,
                    truncation=truncation,
                )
            else:
                return tokenizer.encode(
                    full_text,
                    add_special_tokens=add_special_tokens,
                    return_tensors=return_tensors,
                    padding=padding,
                    truncation=truncation,
                )


class AbsaDatasetLoaderWithConstraints(object):
    """
    This class allows to load a dataset with constraints induced by the models to compare.
    """

    def __init__(
        self,
        root_folder: str,
        name_dataset: str,
        seed: int = 42,
        models: dict = {
            "Prompt": {
                "path_tokenizer": "roberta-base",
                "prompt_template": "<entity> is <mask>.",
            }
        },
        dataset_format: str = "newsmtsc",
    ):
        """Initializes the dataset loader.

        Args:
            root_folder (str): root folder containing all datasets
            name_dataset (str): name of the dataset as key of REGISTERED_DATASETS
            seed (int, optional): seed. Defaults to 42.
            max_len (int, optional): max length tokenizer model. Defaults to 512.
            models (dict, optional): models characteristics. Defaults to {"Prompt":{"path_tokenizer":"roberta-base", "prompt_template":"<entity> is <mask>."}}.
            dataset_format (str, optional): dataset format. Defaults to "newsmtsc".
        """

        self.root_folder = root_folder
        self.name_dataset = name_dataset
        self.seed = seed
        self.models = models
        self.dataset_format = dataset_format
        self.entry_encoder = PromptEncoder()

        # for all models load the tokenizer
        for model in models.values():
            model["tokenizer"] = AutoTokenizer.from_pretrained(model["path_tokenizer"])
            model["valid_indices"] = {}

    def load_dataset(self):
        """Loads the dataset."""
        dataset = REGISTERED_DATASETS[self.name_dataset]
        loader = RawJsonlLoader(self.root_folder, dataset["folder_dataset"])
        data = loader.load_data(dataset["filenames"])
        filtered_dataset = self.constraint_filtering(data)
        splitter = DataSplitter(
            filtered_dataset, dataset["splitting_strategy"], self.seed
        )
        final_data = splitter.get_new_data_splits(
            filtered_dataset, dataset["splitting_strategy"], self.seed
        )
        return final_data

    def load_and_write_dataset(
        self, path_output_folder_dataset, create_if_not_exist=True
    ):
        """Loads the dataset and writes it to a file."""
        dataset = self.load_dataset()
        if not (os.path.exists(path_output_folder_dataset)):
            if create_if_not_exist:
                os.makedirs(path_output_folder_dataset)
            else:
                raise ValueError(
                    "Path {} does not exist".format(path_output_folder_dataset)
                )
        else:
            pass
        mapping_names = ["train.jsonl", "test.jsonl", "validation.jsonl"]
        for k in range(len(dataset)):
            with jsonlines.open(
                os.path.join(path_output_folder_dataset, mapping_names[k]), mode="w"
            ) as writer:
                t = CustomTqdm(
                    dataset[k], desc="Writing dataset {}".format(mapping_names[k])
                )
                for data in t:
                    writer.write(data)

    def load_tokenized_dataset_from_jsonl(
        self, path_output_folder, dataset_to_load="train.jsonl"
    ):
        """
        Loads dataset from a file a tokenize on the fly
        """
        raise NotImplementedError("Not implemented yet")
        mapping = ["train.jsonl", "test.jsonl", "validation.jsonl"]
        dataset = []
        if not (dataset_to_load in mapping):
            raise ValueError("Dataset {} is not supported".format(dataset_to_load))
        else:
            with jsonlines.open(
                os.path.join(path_output_folder, dataset_to_load), mode="r"
            ) as reader:
                t = CustomTqdm(
                    iter(reader), desc="Loading dataset {}".format(dataset_to_load)
                )
                for data in t:
                    dataset.append(self.encode_entry())

    def constraint_filtering(self, data):
        for model in self.models.values():
            tokenizer = model["tokenizer"]
            for name_split, split in data.items():
                model["valid_indices"][name_split] = []
                for id_entry, item in enumerate(split):
                    mention, text = self.get_mention_and_sentence(
                        item, self.dataset_format
                    )
                    prompt_template = model["prompt_template"]
                    encoded_full_text = self.entry_encoder.encode_entry(
                        mention=mention,
                        text=text,
                        model_prompt_template=prompt_template,
                        tokenizer=tokenizer,
                    )
                    if type(encoded_full_text) == list:
                        add = True
                        for elem in encoded_full_text:
                            if len(elem) >= tokenizer.model_max_length - 2:
                                add = False
                                break
                        if add:
                            model["valid_indices"][name_split].append(id_entry)

                    if len(encoded_full_text) <= tokenizer.model_max_length - 2:
                        model["valid_indices"][name_split].append(id_entry)
                model["valid_indices"][name_split] = set(
                    model["valid_indices"][name_split]
                )

        filtered_dataset = {}
        for name_split, split in data.items():
            filtered_dataset[name_split] = []
            for id_entry, item in enumerate(split):
                add = True
                for model in self.models.values():
                    if id_entry not in model["valid_indices"][name_split]:
                        add = False
                        break
                if add:
                    filtered_dataset[name_split].append(item)
        return filtered_dataset

    def get_mention_and_sentence(self, entry, dataset_format, return_chars=False):
        if dataset_format == "newsmtsc":
            id_mention = entry["primary_gid"]
            text = entry["sentence_normalized"]
            targets = entry["targets"]
            mention = None
            for target in targets:
                if target["Input.gid"] == id_mention:
                    mention = target["mention"]
                    chars = {"from": target["from"], "to": target["to"]}
                    break
            if mention is None:
                raise ValueError("Mention not found")
            if return_chars:
                return mention.strip(), text.strip(), chars
            return mention.strip(), text.strip()
