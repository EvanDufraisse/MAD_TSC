#
# Created on Fri Aug 19 2022
#
# Copyright (c) 2022 CEA - LASTI
# Contact: Evan Dufraisse,  evan.dufraisse@cea.fr. All rights reserved.
#
from unittest.util import sorted_list_difference
from tscbench.data.load.datasets import (
    PromptEncoder,
    RawJsonlLoader,
    DataSplitter,
    REGISTERED_DATASETS,
)
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import logging
from typing import List, Dict
import re
import numpy as np
import copy


def closest_match_regex(sentence, mention, original_mention_pos):
    matches = [(x.start(), x.end()) for x in re.finditer(re.escape(mention), sentence)]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        distances = [abs(x[0] - original_mention_pos[0]) for x in matches]
        return matches[distances.index(min(distances))]
    else:
        raise ValueError(
            "No match found for mention {} in sentence {}".format(mention, sentence)
        )


class AbsaModelProcessor(object):
    def __init__(
        self,
        tokenizer_path="",
        prompt_template="",
        sentiment_mapping={2: 0, 4: 1, 6: 2},
        replace_by_main_mention=False,
        replace_by_special_token=None,
        tokenizer=None,
        return_tensors=False,
        soft_cls=False,
    ):
        if tokenizer_path == "" and tokenizer is None:
            raise ValueError("You must provide a tokenizer")
        if replace_by_main_mention and not (replace_by_special_token is None):
            raise ValueError(
                "replace_by_main_mention and replace_by_special_token cannot be both True"
            )

        if tokenizer_path != "":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.sentiment_mapping = sentiment_mapping
        self.replace_by_main_mention = replace_by_main_mention
        self.replace_by_special_token = replace_by_special_token
        if not (replace_by_special_token is None):
            self.tokenizer.add_tokens([replace_by_special_token])
        self.prompt_encoder = PromptEncoder()
        self.return_tensors = return_tensors
        self.soft_cls = soft_cls

    def return_tokens_chars(self, out, chars):
        return (out.char_to_token(chars[0]), out.char_to_token(chars[1] - 1) + 1)

    def process_entry(
        self,
        sentence,
        mentions_pos,
        main_mention,
        sentiment=None,
        all_mentions=None,
        **kwargs,
    ):
        main_mention = main_mention
        mentions_pos = sorted(mentions_pos, key=lambda x: -x[0])
        new_mentions_pos = []
        if self.replace_by_main_mention:
            for start, end in mentions_pos:
                sentence = sentence[:start] + main_mention + sentence[end:]
                new_mentions_pos.append((start, start + len(main_mention)))
        elif self.replace_by_special_token is not None:
            for start, end in mentions_pos:
                sentence = (
                    sentence[:start] + self.replace_by_special_token + sentence[end:]
                )
                new_mentions_pos.append(
                    (start, start + len(self.replace_by_special_token))
                )
                main_mention = self.replace_by_special_token
        elif self.soft_cls:
            for start, end in mentions_pos:
                sentence = (
                    sentence[:start]
                    + " [SOFT_0] "
                    + sentence[start:end].strip()
                    + " [SOFT_1] "
                    + sentence[end:].lstrip()
                )
                new_mentions_pos.append(
                    (
                        start,
                        start
                        + len(
                            " [SOFT_0] " + sentence[start:end].strip() + " [SOFT_1] "
                        ),
                    )
                )
                # new_mentions_pos.append((start, end))
        else:
            new_mentions_pos = mentions_pos

        if type(self.prompt_template) == list:
            prompt_templates = self.prompt_template
        else:
            # print(self.prompt_template)
            prompt_templates = [self.prompt_template]

        out = []
        for prompt_template in prompt_templates:
            if not (self.return_tensors):
                if len(self.prompt_template) > 0:
                    assert self.prompt_encoder is not None

                    out.append(
                        self.prompt_encoder.encode_entry(
                            main_mention,
                            sentence,
                            prompt_template,
                            self.tokenizer,
                            return_tensors=None,
                            return_mention_tokens=True,
                        )
                    )
                else:
                    out.append(
                        self.tokenizer.encode_plus(
                            sentence, add_special_tokens=False, return_tensors=None
                        )
                    )
            else:
                if len(prompt_template) > 0:
                    assert self.prompt_encoder is not None

                    out.append(
                        self.prompt_encoder.encode_entry(
                            main_mention,
                            sentence,
                            prompt_template,
                            self.tokenizer,
                            return_tensors="pt",
                            return_mention_tokens=True,
                            encode_plus=True,
                            padding="max_length",
                            add_special_tokens=True,
                        )
                    )
                else:
                    out.append(
                        self.tokenizer.encode_plus(
                            sentence,
                            add_special_tokens=True,
                            return_tensors="pt",
                            padding="max_length",
                        )
                    )

        mention_tokens = []
        for idx_mention, (start, end) in enumerate(new_mentions_pos):
            try:
                token_start, token_end = self.return_tokens_chars(out[0], (start, end))
                if token_start is None or token_end is None:
                    raise ValueError()
            except:
                # print(main_mention)
                if not (all_mentions is None):
                    start, end = closest_match_regex(
                        sentence, all_mentions[idx_mention], (start, end)
                    )
                else:
                    start, end = closest_match_regex(
                        sentence, main_mention, (start, end)
                    )
                token_start, token_end = self.return_tokens_chars(out[0], (start, end))
            mention_tokens.append((token_start, token_end))
        if sentiment is not None:
            return (
                [o["input_ids"] for o in out],
                mention_tokens,
                new_mentions_pos,
                self.sentiment_mapping[sentiment],
                None,
            )
        else:
            return (
                [o["input_ids"] for o in out],
                mention_tokens,
                new_mentions_pos,
                None,
                None,
            )

    def set_return_tensors(self, value: bool):
        self.return_tensors = value


def get_main_entity_from_newsmtsc_format_entry(
    entry: dict, add_further_mentions: bool = False
):
    """Return characteristics of the main entity in a newsmtsc format entry

    Args:
        entry (dict): newsmtsc format entry

    Returns:
        dict: main entity characteristics (mention pos, main mention, sentiment)
    """
    primary_gid = entry["primary_gid"]
    targets = entry["targets"]
    mention_target = None
    if len(targets) == 1:
        mention_target = targets[0]
    else:
        for target in targets:
            if target["Input.gid"] == primary_gid:
                mention_target = target
                break
    main_mention = mention_target["mention"]
    polarity = mention_target["polarity"]
    chars = [(mention_target["from"], mention_target["to"])]
    main_mention_pos = (mention_target["from"], mention_target["to"])
    all_mentions = [main_mention]
    if "further_mentions" in mention_target and add_further_mentions:
        for f in mention_target["further_mentions"]:
            chars.append((f["from"], f["to"]))
            all_mentions.append(f["mention"])
    return {
        "mentions_pos": chars,
        "all_mentions": all_mentions,
        "main_mention": main_mention,
        "main_mention_pos": main_mention_pos,
        "sentiment": polarity,
    }


def extract_all_data_from_newsmtsc_format_entry(entry):
    sentence = entry["sentence_normalized"]
    main_entity = get_main_entity_from_newsmtsc_format_entry(entry)
    main_entity["sentence"] = sentence
    return main_entity


import bisect


class NonOverlappingIntevals(object):
    """Allows to filter further mentions that overlap

    Args:
        object (_type_): _description_
    """

    def __init__(self, main_mention_pos, l=[]):
        self.l = l
        self.mentions = {}
        self.main_mention_pos = main_mention_pos

    def output_split(self, output, chunk_size=2):
        assert len(output) % chunk_size == 0
        return [
            tuple(output[i : i + chunk_size]) for i in range(0, len(output), chunk_size)
        ]

    def remove_intervals(self, a, b):
        idx_a = bisect.bisect_left(self.l, a)
        idx_b = bisect.bisect_left(self.l, b)
        self.l.pop(idx_b)
        self.l.pop(idx_a)

    def insert_interval(self, a, b, identifier=None):
        if (
            bisect.bisect_right(self.l, a) == bisect.bisect_left(self.l, b)
            and bisect.bisect_right(self.l, a) % 2 == 0
        ):
            idx_insert = bisect.bisect_right(self.l, a)
            self.l.insert(idx_insert, b)
            self.l.insert(idx_insert, a)
            self.mentions[str((a, b))] = identifier
        else:
            collision_a = bisect.bisect_right(self.l, a)
            collision_b = bisect.bisect_left(self.l, b)
            if collision_a == collision_b:
                collision_a -= 1
                collision_b += 1
            if collision_a % 2 == 1:
                collision_a -= 1
            if collision_b % 2 == 1:
                collision_b += 1
            output_splitted = self.output_split(self.l[collision_a:collision_b])
            if self.main_mention_pos in output_splitted:
                return
            if len(output_splitted) > 1:
                for elem in output_splitted:
                    self.remove_intervals(*elem)
                    del self.mentions[str(elem)]
                self.insert_interval(a, b, identifier)
            elif len(output_splitted) == 1:
                elem = output_splitted[0]
                if elem[0] > a or elem[1] < b:
                    self.remove_intervals(*elem)
                    del self.mentions[str(elem)]
                    self.insert_interval(a, b, identifier)
            else:
                pass


def duplicate_entry_with_all_targets(entry):
    entries = []
    for target in entry["targets"]:
        entry["primary_gid"] = target["Input.gid"]
        entries.append(entry.copy())
    return entries


import json


def unicity_filtering_further_mentions(entry):
    """Check that main mention isn't in further mentions, and that no further mention overlaps another

    Args:
        entry (_type_): _description_

    Returns:
        _type_: _description_
    """
    for target in entry["targets"]:
        if target["Input.gid"] == entry["primary_gid"]:
            main_from = target["from"]
            main_to = target["to"]
            if "further_mentions" in target:
                set_further_mentions = set(
                    [json.dumps(e) for e in target["further_mentions"]]
                )
                unique_further_mentions = [json.loads(e) for e in set_further_mentions]
                kept_further_mentions = [
                    mention
                    for mention in unique_further_mentions
                    if (mention["from"] >= main_to or mention["to"] <= main_from)
                ]
                if len(kept_further_mentions) > 0:
                    target["further_mentions"] = kept_further_mentions
                else:
                    del target["further_mentions"]
            if "further_mentions" in target:
                no_overlap_further_mentions = NonOverlappingIntevals(
                    (main_from, main_to), []
                )
                no_overlap_further_mentions.insert_interval(
                    main_from, main_to, "main_mention"
                )
                for idx_further_mention, mention in enumerate(
                    target["further_mentions"]
                ):
                    no_overlap_further_mentions.insert_interval(
                        mention["from"], mention["to"], idx_further_mention
                    )
                kept_mentions_idx = set(no_overlap_further_mentions.mentions.values())
                new_further_mentions = []
                for idx_further_mention, mention in enumerate(
                    target["further_mentions"]
                ):
                    if idx_further_mention in kept_mentions_idx:
                        new_further_mentions.append(mention)
                target["further_mentions"] = new_further_mentions
            break
        else:
            continue
    return entry


class AbsaDatasetConstraintsFiltering(object):
    """
    This class allows to load a dataset with constraints induced by the models to compare.
    """

    def __init__(
        self,
        root_folder: str,
        name_dataset: str,
        models_processors: List[AbsaModelProcessor],
        seed=42,
        split_before_filtering=True,
        path_dataset=None,
    ):
        """
        Filter the dataset based on constraints over all models processors
        """
        self.name_dataset = name_dataset
        self.root_folder = root_folder
        self.models_processors = models_processors
        self.seed = seed
        self.split_before_filtering = split_before_filtering
        self.path_dataset = path_dataset

    def constraint_filtering(self, dataset_config=None, no_split=False):
        sort_and_reverse = SortAndReverse()
        counter = 0
        total = 0
        if dataset_config is None:
            if self.path_dataset is not None:
                dataset_config = {
                    "name_dataset": self.name_dataset,
                    "format": "newsmtsc",
                    "folder_dataset": self.path_dataset.split("/")[-2],
                    "filenames": {
                        self.path_dataset.split("/")[-1].split(".")[
                            0
                        ]: self.path_dataset.split("/")[-1],
                    },
                    "splitting_strategy": {
                        self.path_dataset.split("/")[-1].split(".")[0]: [1],
                    },
                }
                self.root_folder = "/".join(self.path_dataset.split("/")[:-2])
            else:
                raise NotImplemented("No dataset config provided")
                # dataset = REGISTERED_DATASETS[self.name_dataset]
        loader = RawJsonlLoader(self.root_folder, dataset_config["folder_dataset"])
        loaded_datasets = loader.load_data(dataset_config["filenames"])
        splitting_strategy = dataset_config["splitting_strategy"]
        if self.split_before_filtering:
            splitter = DataSplitter(loaded_datasets, splitting_strategy, self.seed)

            loaded_datasets = splitter.get_new_data_splits(
                loaded_datasets, dataset_config["splitting_strategy"], self.seed
            )
            if self.path_dataset is not None:
                new_loaded_datasets = {
                    split_name: loaded_datasets[idx_split]
                    for idx_split, split_name in enumerate(
                        list(splitting_strategy.keys())
                    )
                }
                loaded_datasets = new_loaded_datasets
            else:
                loaded_datasets = {
                    "train": loaded_datasets[0],
                    "validation": loaded_datasets[2],
                    "test": loaded_datasets[1],
                }
            if dataset_config is not None and dataset_config["name_dataset"] in [
                "newsmtscmt",
                "newsmtscrw",
            ]:
                # print("Duplicating newsmtsc dataset")
                new_loaded_datasets = {}
                for split, dataset in loaded_datasets.items():
                    new_loaded_datasets[split] = []
                    for entry in dataset:
                        duplicated_entries = duplicate_entry_with_all_targets(entry)
                        further_mentions_filtered_entries = []
                        for duplicated_entry in duplicated_entries:
                            further_mentions_filtered_entries.append(
                                unicity_filtering_further_mentions(duplicated_entry)
                            )
                        new_loaded_datasets[split] += further_mentions_filtered_entries
                loaded_datasets = new_loaded_datasets
            else:
                for key, datasplit in loaded_datasets.items():
                    for idx_entry, entry in enumerate(datasplit):
                        datasplit[idx_entry] = unicity_filtering_further_mentions(entry)
        final_data = {}
        for key, d in loaded_datasets.items():
            temp_data = []
            for entry in tqdm(d):
                data_entry = extract_all_data_from_newsmtsc_format_entry(entry)
                overflow = False
                total += 1
                all_mentions = data_entry["all_mentions"]
                for model_processor in self.models_processors:
                    try:
                        (
                            list_tokens,
                            mention_tokens,
                            mentions_pos,
                            sentiment,
                            params,
                        ) = model_processor.process_entry(
                            data_entry["sentence"],
                            data_entry["mentions_pos"],
                            data_entry["main_mention"],
                            data_entry["sentiment"],
                            all_mentions=all_mentions,
                        )
                    except:
                        return entry
                    for tokens in list_tokens:
                        if len(tokens) > model_processor.tokenizer.model_max_length - 2:
                            overflow = True
                            break
                    if overflow:
                        break
                if not overflow:
                    entry["max_len_tokens"] = max([len(elem) for elem in list_tokens])
                    for (
                        key_characteristics,
                        value_key_characteristics,
                    ) in get_main_entity_from_newsmtsc_format_entry(entry).items():
                        entry[key_characteristics] = value_key_characteristics

                    sort_and_reverse.sort(entry["mentions_pos"], lambda x: -x[0])
                    entry["processed_mention_tokens"] = sort_and_reverse.reverse(
                        mention_tokens
                    )
                    entry["processed_mentions_pos"] = sort_and_reverse.reverse(
                        mentions_pos
                    )
                    temp_data.append(entry)
                else:
                    counter += 1
                    continue
            final_data[key] = temp_data
        logging.warning(f"{counter} over {total} entries were filtered")
        if not (self.split_before_filtering) and not (no_split):
            splitter = DataSplitter(final_data, splitting_strategy, self.seed)
            final_data = splitter.get_new_data_splits(
                final_data, dataset_config["splitting_strategy"], self.seed
            )

        return final_data


import torch


class DummyValue(object):
    def __init__(self, value):
        self.value = value

    def __getitem__(self, idx):
        return self.value


class AbsaDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, mentions_tokens, mentions_pos, sentiments, params=None):
        self.tokens = tokens
        self.mentions_tokens = mentions_tokens
        self.mentions_pos = mentions_pos
        if len(sentiments) > 0:
            self.sentiments = sentiments
        else:
            self.sentiments = DummyValue(None)
        if len(params) > 0:
            self.params = params
        else:
            self.params = DummyValue(None)

    def __getitem__(self, idx):
        return (
            self.tokens[idx],
            self.mentions_tokens[idx],
            self.mentions_pos[idx],
            self.sentiments[idx],
            self.params[idx],
        )

    def __len__(self):
        return len(self.tokens)


class AbsaDatasetLoader(object):
    def __init__(self, data, absa_model_processor: AbsaModelProcessor):
        self.data = data
        self.absa_model_processor = absa_model_processor

    def load_data(self, specific_split=None, shuffle=False, prob_shuffle=None):
        datasets = {}
        for key, subset in self.data.items():
            if specific_split is not None and shuffle:
                if key != specific_split:
                    continue
                else:
                    # print(type(subset))
                    # print(len(subset))
                    # print("Loading only split", specific_split)
                    if prob_shuffle is None:
                        mention_replace = ReplaceMentionEntry(subset)
                    else:
                        mention_replace = ReplaceMentionEntry(
                            subset, prob_replace=prob_shuffle
                        )
                    new_subset = []
                    for entry in subset:
                        new_subset.append(mention_replace.replace_mention_entry(entry))
                    subset = new_subset
                    # print("Number of entries", len(subset))
            all_tokens = []
            all_mentions_tokens = []
            all_mentions_pos = []
            all_sentiments = []
            all_params = []
            for entry in subset:
                data_entry = extract_all_data_from_newsmtsc_format_entry(entry)
                try:
                    (
                        tokens,
                        mention_tokens,
                        mentions_pos,
                        sentiment,
                        params,
                    ) = self.absa_model_processor.process_entry(**data_entry)
                except Exception as e:
                    # print(e)
                    # print(data_entry)
                    return entry
                all_tokens.append(tokens)
                all_mentions_tokens.append(mention_tokens)
                all_mentions_pos.append(mentions_pos)
                if sentiment is not None:
                    all_sentiments.append(sentiment)
                if params is not None:
                    all_params.append(params)
            datasets[key] = AbsaDataset(
                all_tokens,
                all_mentions_tokens,
                all_mentions_pos,
                all_sentiments,
                all_params,
            )
        return datasets


class AbsaDataCollator(object):
    def __init__(
        self,
        mode_mask,
        tokenizer_mask_id,
        tokenizer_padding_id,
        force_gpu=False,
    ):
        self.mode_mask = mode_mask
        self.tokenizer_mask_id = tokenizer_mask_id
        self.force_gpu = force_gpu
        self.tokenizer_padding_id = tokenizer_padding_id

    def __call__(self, batch):
        attention_masks = []
        batch_tokens = []
        sentiments = []
        params = []
        for entry in batch:
            batch_tokens += entry[0]
            sentiments += [
                entry[-2]
            ]  # *len(entry[0]) la fusion est faite dans le modèle
            params += [entry[-1]] * len(entry[0])
        batch_tokens = torch.concat(batch_tokens, dim=0)
        attention_masks = torch.zeros_like(batch_tokens)
        attention_masks[batch_tokens != self.tokenizer_padding_id] = 1
        if self.mode_mask:
            classifying_locations = torch.where(batch_tokens == self.tokenizer_mask_id)
        else:
            classifying_locations = []
            classifying_positions_x = []
            classifying_positions_y = []
            for idx, sample in enumerate(batch):
                temp_y = []
                for token_pos in sample[1]:
                    temp_y += [i for i in range(token_pos[0], token_pos[1])]
                classifying_positions_x += [idx] * len(temp_y)
                classifying_positions_y += temp_y
            classifying_locations = torch.tensor(classifying_positions_x), torch.tensor(
                classifying_positions_y
            )
        counts = torch.bincount(classifying_locations[0])
        x_select = torch.zeros(batch_tokens.shape)
        x_select[classifying_locations] = 1
        if params[0] is not None:
            params = torch.tensor(params)
        else:
            params = torch.tensor([-1])
        if sentiments[0] is not None:
            sentiments = torch.tensor(sentiments)
        else:
            sentiments = torch.tensor([-1])
        if not (self.force_gpu):
            return (
                batch_tokens,
                attention_masks,
                x_select,
                classifying_locations,
                counts,
                sentiments,
                params,
            )
        else:
            return (
                batch_tokens.to("cuda"),
                attention_masks.to("cuda"),
                x_select.to("cuda"),
                (
                    classifying_locations[0].to("cuda"),
                    classifying_locations[1].to("cuda"),
                ),
                counts.to("cuda"),
                sentiments,
                params,
            )


class ReplaceMentionEntry(object):
    def __init__(self, dataset, prob_replace=0.5, ex_limit_size_subword=3):
        self.dataset = dataset
        self.prob_replace = prob_replace
        self.all_main_mentions = self.extract_all_main_mentions(dataset)
        self.ex_limit_size_subword = ex_limit_size_subword

    def extract_all_main_mentions(self, dataset):
        """Extracts all main mention from a dataset split, and returns a dictionary with the number of tokens as key and a list of mentions as value

        Args:
            dataset (dict[str,dict]): dataset that has been filtered and processed by the model_processor (non tokenized dataset)

        Returns:
            dict[int, list[str]]: dictionnary to sample new mention from
        """
        all_main_mentions = {}
        for entry in dataset:
            output = get_main_entity_from_newsmtsc_format_entry(entry)
            main_mention = output["main_mention"]
            main_mention_len = len(main_mention.split(" "))
            main_mention_n_tokens = (
                entry["processed_mention_tokens"][0][1]
                - entry["processed_mention_tokens"][0][0]
            )
            if main_mention_len < 6:
                if main_mention_n_tokens in all_main_mentions:
                    all_main_mentions[main_mention_n_tokens].add(main_mention)
                else:
                    all_main_mentions[main_mention_n_tokens] = set([main_mention])
        for key, set_mentions in all_main_mentions.items():
            all_main_mentions[key] = list(set_mentions)
        return all_main_mentions

    def get_subwords_name(self, name, limit_size=None):
        """Returns a list of subwords of a name if subword is longer than limit_size, only alpha characters and lowercase"""
        if limit_size is None:
            limit_size = self.ex_limit_size_subword
        subwords = []
        for subword in name.split(" "):
            sw = "".join([c.lower() for c in subword if c.isalpha()])
            if len(sw) > limit_size:
                subwords.append(sw)
        return subwords

    def get_further_mentions_with_overlapping_subnames(self, entry):
        """Returns the indices of further mentions that have a subname that is in the main mention,
        those mentions should also be replaced to maintain a coherent sentence.

        Args:
            entry (dict): classic non tokenized entry

        Returns:
            list[int]: list of indices of further mentions to replace
        """
        output = get_main_entity_from_newsmtsc_format_entry(entry)
        main_mention = output["main_mention"]
        idx_to_replace = []
        subwords_main_mention = set(self.get_subwords_name(main_mention))
        for i in range(1, len(output["all_mentions"])):
            subwords_further_mention = set(
                self.get_subwords_name(output["all_mentions"][i])
            )
            if len(subwords_main_mention.intersection(subwords_further_mention)) > 0:
                idx_to_replace.append(i)
        return idx_to_replace

    def random_sampling_all_main_mentions(self, all_main_mentions, limit_n_tokens):
        """Randomly samples a mention from the all_main_mentions dictionary and returns it"""
        sum_mentions = sum(
            [len(x) for key, x in all_main_mentions.items() if key <= limit_n_tokens]
        )
        prob = [
            len(x) / sum_mentions
            for key, x in all_main_mentions.items()
            if key <= limit_n_tokens
        ]
        sampling = np.random.multinomial(1, prob, size=1)[0]
        idx_key = np.argmax(sampling)
        key = list(all_main_mentions.keys())[idx_key]
        return all_main_mentions[key][np.random.randint(len(all_main_mentions[key]))]

    def replace_mention_entry(self, entry, all_main_mentions=None, prob_replace=None):
        e = copy.deepcopy(entry)
        if all_main_mentions is None:
            all_main_mentions = self.all_main_mentions
        if prob_replace is None:
            prob_replace = self.prob_replace
        if np.random.rand() < prob_replace:
            idx_to_replace = [0]
            idx_to_replace += self.get_further_mentions_with_overlapping_subnames(e)
            n_tokens_to_replace = 0

            mentions_pos = []
            for idx in idx_to_replace:
                n_tokens_to_replace += (
                    e["processed_mention_tokens"][idx][1]
                    - e["processed_mention_tokens"][idx][0]
                )
                # mentions_pos.append(e["processed_mentions_pos"][idx])
            max_token_len = (512 - 2 - n_tokens_to_replace) // len(idx_to_replace)
            new_mention = self.random_sampling_all_main_mentions(
                all_main_mentions, max_token_len
            )
            e["sentence_normalized"] = self._replace_mention_in_sentence(
                e, e["processed_mentions_pos"], new_mention, idx_to_replace
            )
        return e

    def _replace_mention_in_sentence(
        self, entry, mentions_pos, new_mention, idx_to_replace
    ):
        new_sentence = entry["sentence_normalized"]
        main_mention_pos = mentions_pos[0]
        # print(main_mention_pos)
        sort_and_reverse = SortAndReverse()
        mentions_pos_sorted = sort_and_reverse.sort(
            mentions_pos, sorting_key=lambda x: x[0]
        )
        order = sort_and_reverse.last_pos
        index_main_mention_pos = mentions_pos_sorted.index(main_mention_pos)
        # print(index_main_mention_pos)
        # print(main_mention_pos)
        # print(mentions_pos_sorted)
        offset = 0
        new_pos_sorted = []
        # print(new_mention)
        # print(new_sentence[mentions_pos_sorted[0][0]:mentions_pos_sorted[0][1]])
        for temp_idx, pos in enumerate(mentions_pos_sorted):
            if order[temp_idx] in idx_to_replace:
                new_sentence = (
                    new_sentence[: pos[0] + offset]
                    + new_mention
                    + new_sentence[pos[1] + offset :]
                )
                new_pos_sorted.append(
                    (pos[0] + offset, pos[0] + offset + len(new_mention))
                )
                offset += len(new_mention) - (pos[1] - pos[0])
            else:
                new_pos_sorted.append((pos[0] + offset, pos[1] + offset))
        main_target = self.get_main_target(entry)
        main_target["from"] = new_pos_sorted[index_main_mention_pos][0]
        main_target["to"] = new_pos_sorted[index_main_mention_pos][1]
        main_target["mention"] = new_mention
        if "further_mentions" in main_target:
            for idx, further_mention in enumerate(main_target["further_mentions"]):
                if idx + 1 in idx_to_replace:
                    # print("idx_to_replace")
                    idx_further = mentions_pos_sorted.index(
                        (further_mention["from"], further_mention["to"])
                    )
                    # print(idx_further)
                    further_mention["from"] = new_pos_sorted[idx_further][0]
                    further_mention["to"] = new_pos_sorted[idx_further][1]
                    further_mention["mention"] = new_mention
                else:
                    idx_further = mentions_pos_sorted.index(
                        (further_mention["from"], further_mention["to"])
                    )
                    # print(idx_further)
                    further_mention["from"] = new_pos_sorted[idx_further][0]
                    further_mention["to"] = new_pos_sorted[idx_further][1]

        return new_sentence

    def get_main_target(self, entry):
        for target in entry["targets"]:
            if target["Input.gid"] == entry["primary_gid"]:
                return target


class SortAndReverse(object):
    def __init__(self):
        self.last_pos = None

    def reverse(self, l):
        if self.last_pos is None:
            raise ValueError("You must call sort before reverse")
        else:
            l2 = zip(self.last_pos, l)
            l2 = sorted(l2, key=lambda x: x[0])
            return [x[1] for x in l2]

    def sort(self, l, sorting_key=lambda x: x):
        source_indices = [k for k in range(len(l))]
        l2 = list(zip(source_indices, l))
        l2.sort(key=lambda x: sorting_key(x[1]))
        self.last_pos = [x[0] for x in l2]
        return [x[1] for x in l2]
