#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Sampler module for choice of hyperparameters

@Author: Evan Dufraisse
@Date: Wed May 03 2023
@Contact: evan[dot]dufraisse[at]cea[dot]fr
@License: Copyright (c) 2023 CEA - LASTI
"""
import sqlite3
import os


class Sampler(object):
    def __init__(self, path_to_db):
        self.path_to_db = path_to_db
        if not (os.path.exists(self.path_to_db)):
            self.create_db()

    def create_db(self, hyperparameters):
        conn = sqlite3.connect(self.path_to_db)
        c = conn.cursor()
        # Lock the database
        c.execute("BEGIN EXCLUSIVE")
        # Determine types of hyperparameters
        types = {}
        for k, v in hyperparameters.items():
            if isinstance(v[0], int):
                types[k] = "INTEGER"
            elif isinstance(v[0], float):
                types[k] = "REAL"
            elif isinstance(v[0], str):
                types[k] = "TEXT"
            else:
                raise ValueError("Type of hyperparameter {} not supported".format(k))
        # Create table
        c.execute(
            "CREATE TABLE hyperparameters ("
            + ", ".join(["{} {}".format(k, v) for k, v in types.items()])
            + ", status TEXT, RESULT REAL)"
        )
        # Commit changes
        conn.commit()
        # Unlock the database
        c.execute("END")
        conn.close()
