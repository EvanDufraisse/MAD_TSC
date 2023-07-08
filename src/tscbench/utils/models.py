#
# Created on Wed Aug 03 2022
#
# Copyright (c) 2022 CEA - LASTI
# Contact: Evan Dufraisse,  evan.dufraisse@cea.fr. All rights reserved.
#
import torch
from collections import OrderedDict
import re
import logging


class ModelManager(object):
    """Allows to save and load a model from a checkpoint, even if state_dict do not fully match.

    Args:
        object (_type_): _description_
    """

    def __init__(self):
        pass

    def get_mapping_match_state_keys(self, original_model_keys, new_model_keys):
        """
        Returns a mapping of the keys in the new model to the keys in the original model.
        """
        mapping = {}
        for key in original_model_keys:
            for new_key in new_model_keys:
                first_match = re.search(re.escape(key), new_key)
                if not (first_match is None):
                    extract_new_model_key = new_key[first_match.span()[0] :]
                    if re.search(re.escape(extract_new_model_key), key):
                        mapping[key] = new_key
        return mapping

    def load_model_from_ckpt(self, model, ckpt_path):
        """Given a model and a checkpoint path, load the model from the checkpoint.

        Args:
            model (model): _description_
            ckpt_path (str): _description_

        Returns:
            model: _description_
        """
        ckpt_checkpoint = torch.load(ckpt_path)
        return self.load_model_from_state_dict(model, ckpt_checkpoint["state_dict"])

    def load_model_from_state_dict(self, model, state_dict):
        new_model_state_dict = state_dict
        new_state_dict = OrderedDict()
        original_state_dict = model.state_dict()
        mapping_state_dict = self.get_mapping_match_state_keys(
            original_state_dict.keys(), new_model_state_dict.keys()
        )
        for key in original_state_dict.keys():
            new_state_dict[key] = new_model_state_dict[mapping_state_dict[key]]
        model.load_state_dict(new_state_dict)

        return model

    def save_model_to_ckpt(self, model, ckpt_path, params={}):
        """Given a model and a checkpoint path, save the model to the checkpoint.

        Args:
            model (model): _description_
            ckpt_path (str): _description_
        """
        torch.save({"state_dict": model.state_dict(), "params": params}, ckpt_path)
