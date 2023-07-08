# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = {"": "src"}

packages = ["tscbench"]

package_data = {"": ["*"]}

install_requires = [
    "transformers",
    "torch",
    "tqdm",
    "jsonlines",
    "numpy",
    "loguru",
    "click",
    "requests",
    "ipython",
    "pytorch_lightning",
    "names_generator",
    "scikit_learn",
    "optuna",
    "dateparser",
    "tensorboard",
]

dependency_links = []

entry_points = {"console_scripts": ["tscbench = tscbench.cli:cli"]}

setup_kwargs = {
    "name": "TscBench",
    "version": "0.1.0",
    "description": "",
    "long_description": "# TSC benchmark package",
    "author": "Evan Dufraisse",
    "author_email": "edufraisse@gmail.com",
    "maintainer": "None",
    "maintainer_email": "None",
    "url": "None",
    "package_dir": package_dir,
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "entry_points": entry_points,
    "dependency_links": dependency_links,
    "python_requires": ">=3.7",
}


setup(**setup_kwargs)
