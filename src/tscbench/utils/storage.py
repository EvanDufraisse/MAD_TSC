# -*- coding: utf-8 -*-
""" Storage manager for experiments

@Author: Evan Dufraisse
@Date: Thu May 04 2023
@Contact: evan[dot]dufraisse[at]cea[dot]fr
@License: Copyright (c) 2023 CEA - LASTI
"""

import os
from datetime import datetime
import random
import shutil
import logging


def generate_uuid(export=False):
    uuid = datetime.today().strftime("%y_%m_%d_%H_%M_%S") + str(random.randint(0, 9999))
    if export:
        os.environ["UUID"] = uuid
    return uuid


class ExperimentStorageManager(object):
    def __init__(
        self,
        sub_path_final_folder,
        uuid=None,
        experiment_dir=None,
        data_dir=None,
        experiment_scratch=None,
        data_scratch=None,
    ):
        """

        Generates unique folders for the experiment depending on the environment variables
        $DATA_DIR
        $DATA_SCRATCH
        $EXPERIMENT_DIR
        $EXPERIMENT_SCRATCH

        Args:
            sub_path_final_folder (str): ${EXPERIMENT_DIR}sub_path_final_folder is the final folder
            uuid (str, optional): unique identifier for the experiment. Defaults to None.

        Raises:
            ValueError: if unique identifier environment variable is not defined and not supplied
            ValueError: if all folder environment variables aren't defined
        """
        if "UUID" not in os.environ and uuid is None:
            raise ValueError("UUID environment variable not set")

        mapping_name_to_variable = {
            "EXPERIMENT_DIR": experiment_dir,
            "DATA_DIR": data_dir,
            "DATA_SCRATCH": data_scratch,
            "EXPERIMENT_SCRATCH": experiment_scratch,
        }

        for key in mapping_name_to_variable.keys():
            if mapping_name_to_variable[key] is not None:
                setattr(self, key.lower(), mapping_name_to_variable[key])
                continue
            elif not (key in os.environ):
                raise ValueError(f"{key} environment variable not set")
            else:
                setattr(self, key.lower(), os.environ[key])
        if uuid is None:
            self.uuid = os.environ["UUID"]
        else:
            self.uuid = uuid
        # self.experiment_dir = os.environ["EXPERIMENT_DIR"]
        # self.data_dir = os.environ["DATA_DIR"]
        # self.experiment_scratch = os.environ["EXPERIMENT_SCRATCH"]
        # self.data_scratch = os.environ["DATA_SCRATCH"]
        if sub_path_final_folder.startswith("/"):
            sub_path_final_folder = sub_path_final_folder[1:]
        self.sub_path_final_folder = sub_path_final_folder.strip("/")

    def generate_folders(self):
        """Generate folders structure for an experiment.
        Suppose the existence of an UUID environment variable to generate a unique folder for the experiment.
        """
        if not (os.path.isdir(self.experiment_scratch)):
            os.makedirs(self.experiment_scratch)
        if not (os.path.isdir(self.data_scratch)):
            os.makedirs(self.data_scratch)

        if os.path.isdir(os.path.join(self.data_scratch, self.uuid)):
            # raise ValueError(f"Experiment {self.uuid} already exists")
            pass
        else:
            os.makedirs(os.path.join(self.data_scratch, self.uuid))

        if os.path.isdir(os.path.join(self.experiment_scratch, self.uuid)):
            pass
            # raise ValueError(f"Experiment {self.uuid} already exists")
        else:
            r = os.path.join(self.experiment_scratch, self.uuid)
            os.makedirs(os.path.join(r, "scratch"))
            os.makedirs(os.path.join(r, "scratch_temp"))

    def delete_scratch_folders(self):
        """Delete temporary scratch folder."""
        if os.path.isdir(os.path.join(self.data_scratch, self.uuid)):
            shutil.rmtree(os.path.join(self.data_scratch, self.uuid))
        if os.path.isdir(os.path.join(self.experiment_scratch, self.uuid)):
            shutil.rmtree(os.path.join(self.experiment_scratch, self.uuid))

    def delete_temp_scratch_folder(self):
        """Delete temporary scratch folder."""
        if os.path.isdir(
            os.path.join(self.experiment_scratch, self.uuid, "scratch_temp")
        ):
            shutil.rmtree(
                os.path.join(self.experiment_scratch, self.uuid, "scratch_temp")
            )

    def transfer_scratch_to_final(self):
        """Transfer temporary scratch folder to final experiment folder."""
        if os.path.isdir(os.path.join(self.experiment_scratch, self.uuid, "scratch")):
            os.makedirs(
                os.path.join(
                    self.experiment_dir,
                    (self.sub_path_final_folder + "//").replace("//", "/"),
                ),
                exist_ok=True,
            )
            command = f'rsync -avh --progress {os.path.join(self.experiment_scratch, self.uuid, "scratch/")} {os.path.join(self.experiment_dir, (self.sub_path_final_folder+"/").replace("//", "/"))}'
            logging.critical(command)
            os.system(command)

    def get_path(self, sub_path: str, choice: str):
        """
        choices: "dscratch", "escratch", "etemp", "final"
        """
        if sub_path.startswith("/"):
            sub_path = sub_path[1:]
        if choice == "dscratch":
            return os.path.join(self.data_scratch, self.uuid, sub_path)
        elif choice == "escratch":
            return os.path.join(self.experiment_scratch, self.uuid, "scratch", sub_path)
        elif choice == "etemp":
            return os.path.join(
                self.experiment_scratch, self.uuid, "scratch_temp", sub_path
            )
        elif choice == "final":
            return os.path.join(
                self.experiment_dir, self.sub_path_final_folder, sub_path
            )
