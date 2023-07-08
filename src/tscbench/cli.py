#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Command line Menu for TscBench

@Author: Evan Dufraisse
@Date: Thu Jun 29 2023
@Contact: evan[dot]dufraisse[at]cea[dot]fr
@License: Copyright (c) 2023 CEA - LASTI
"""
import sys
import click
from loguru import logger
from tscbench.commands.finetuning import cli_finetune


@click.group()
@click.option("--debug", is_flag=True)
def cli(debug):
    if not debug:
        logger.remove()
        logger.add(sys.stderr, level="INFO")


cli.add_command(cli_finetune)
