# -*- coding: utf-8 -*-
""" Set of functions

@Author: Evan Dufraisse
@Date: Thu May 04 2023
@Contact: evan[dot]dufraisse[at]cea[dot]fr
@License: Copyright (c) 2023 CEA - LASTI
"""


def find_highest_divisor(target_effective_batch_size, low, high):
    high = min(high, target_effective_batch_size)
    for k in range(high, low - 1, -1):
        if target_effective_batch_size % k == 0:
            return k
        else:
            continue
