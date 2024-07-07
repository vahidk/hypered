#!/usr/bin/env python3

"""Hyperparameter optimizer.

This script runs a hyperparameter optimization process based on a given configuration file.

Usage:
    hypered.py config

Args:
    config (str): The path to the configuration file.
"""

import argparse
import logging

from . import interface  # noqa: F401
from .interface.registry import get_symbols


def main():
    """Main function to run the hyperparameter optimization."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Hyperparameter optimizer.")
    parser.add_argument("config", type=str, help="Configuration file path.")
    args = parser.parse_args()

    cfg = open(args.config).read()
    eval(cfg, get_symbols())


if __name__ == "__main__":
    main()
