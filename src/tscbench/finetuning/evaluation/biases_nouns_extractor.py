# -*- coding: utf-8 -*-
""" BiasesCommonNounsExtractor

Extracts the polarity biases of NewsMTSC format datasets.

@Author: Evan Dufraisse
@Date: Fri Nov 25 2022
@Contact: evan[dot]dufraisse[at]cea[dot]fr
@License: Copyright (c) 2022 CEA - LASTI
"""
import os
import json
import jsonlines
from tqdm.auto import tqdm
import numpy as np


class BiasesCommonNounsExtractor(object):
    def __init__(
        self, path_root_datasets=None, datasets_files=None, dataset_entries=None
    ):
        """Initialisation of the biases extractor.

        Args:
            path_root_datasets (str): root path where are located the jsonl files of the datasets
            datasets_files (list[str], optional): list of specific files present within the root folder . Defaults to None. If None, all jsonl are considered.
        """
        if path_root_datasets is None and dataset_entries is None:
            raise ValueError(
                "Either path_root_datasets or dataset_entries must be provided."
            )
        self.path_root_datasets = path_root_datasets
        self.datasets_files = datasets_files
        self.dataset_entries = dataset_entries
        if path_root_datasets is not None:
            if datasets_files is None:
                self.datasets_files = [
                    f for f in os.listdir(path_root_datasets) if f.endswith(".jsonl")
                ]
            else:
                self.datasets_files = datasets_files

            self.assert_path_exists()
        self.biases = None

    def assert_path_exists(self):
        """Assert that the path to the datasets exists."""
        for file in self.datasets_files:
            assert os.path.exists(
                os.path.join(self.path_root_datasets, file)
            ), f"File {file} does not exist."

    def extract_biases(self):
        """Extract the biases from the datasets."""
        self.biases = {}
        if self.dataset_entries is None:
            for file in self.datasets_files:
                # name_file = file.split("/")[-1].split(".")[0]
                # self.biases[name_file] = {}
                self.biases = {}
                for line in tqdm(
                    jsonlines.open(os.path.join(self.path_root_datasets, file))
                ):
                    for target in line["targets"]:
                        mention = target["mention"]
                        polarity = target["polarity"]
                        if mention not in self.biases:
                            self.biases[mention] = []
                        self.biases[mention].append(polarity)
        else:
            for line in tqdm(self.dataset_entries):
                for target in line["targets"]:
                    mention = target["mention"]
                    polarity = target["polarity"]
                    if mention not in self.biases:
                        self.biases[mention] = []
                    self.biases[mention].append(polarity)

    def return_sorted_by_occurences(self, min_occ=10):
        """Return the biases sorted by occurences."""
        if self.biases is None:
            self.extract_biases()
        self.sorted_biases = [
            {
                "mention": key,
                "occurences": len(value),
                "avg_polarity": np.mean(value),
                "std_polarity": np.std(value),
                "q1 polarity": np.percentile(value, 25),
                "median polarity": np.median(value),
                "q3 polarity": np.percentile(value, 75),
            }
            for key, value in self.biases.items()
            if len(value) >= min_occ
        ]
        self.sorted_biases.sort(key=lambda x: x["occurences"], reverse=True)
        return self.sorted_biases

    def return_identical_percentile(self, min_occ=10):
        return [
            elem
            for elem in self.return_sorted_by_occurences()
            if elem["q1 polarity"] == elem["q3 polarity"]
            and elem["occurences"] >= min_occ
        ]
