#!/usr/bin/env python3

"""Hyperparameter optimizer.

This script runs a hyperparameter optimization process based on a given configuration file.

Usage:
    hypered.py config

Args:
    config (str): The path to the configuration file.
"""

import argparse

from . import interface
from .interface.common import registry


def main():
    """Main function to run the hyperparameter optimization."""
    parser = argparse.ArgumentParser(description="Hyperparameter optimizer.")
    parser.add_argument("config", type=str, help="Configuration file path.")
    args = parser.parse_args()

    cfg = open(args.config).read()
    eval(cfg, registry.get_symbols())


if __name__ == "__main__":
    main()
