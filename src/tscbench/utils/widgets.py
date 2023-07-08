# -*- coding: utf-8 -*-
""" Widgets for the GUI

@Author: Evan Dufraisse
@Date: Thu Apr 27 2023
@Contact: evan[dot]dufraisse[at]cea[dot]fr
@License: Copyright (c) 2023 CEA - LASTI
"""
from tqdm.auto import tqdm


class CustomTqdm(object):
    """
    tqdm wrapper that allows easy tracking of a dictionary of stats
    """

    def __init__(
        self,
        iterator,
        update_every=1,
        dict_values={},
        unit="items",
        desc="",
        enumeration=False,
        total=None,
    ):
        self.iterator = iterator
        self.update_every = update_every
        self.i = 0
        self.dict_values = dict_values
        self.postfix = self.get_postfix_str()
        self.unit = unit
        self.desc = desc
        self.enumeration = enumeration
        self.total = total

        if self.enumeration:
            self.t = tqdm(
                enumerate(iterator),
                postfix=self.postfix,
                desc=self.desc,
                unit=self.unit,
                total=self.total,
            )
        else:
            self.t = tqdm(
                iterator,
                postfix=self.postfix,
                desc=self.desc,
                unit=self.unit,
                total=self.total,
            )

    def update(self):
        if self.i > self.update_every:
            self.i = 0
            self.t.update()
        else:
            self.i += 1

    def force_update(self):
        self.t.update()
        self.i = 0

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.t.__exit__(*args, **kwargs)

    def __iter__(self):
        return iter(self.t)

    def set_postfix(self, dict_values):
        new_postfix = []
        self.dict_values = dict_values
        self.t.postfix = self.get_postfix_str()

    def get_postfix_str(self):
        new_postfix = []
        for key, value in self.dict_values.items():
            new_postfix.append(f" {key}:{value}")
        new_str_postfix = " |".join(new_postfix)
        new_str_postfix = "|" + new_str_postfix
        return new_str_postfix
