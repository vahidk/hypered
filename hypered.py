"""Hyper parameter optimizer.

Usage:
    hypered.py config_file
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import interface

from interface import registry


def main(args):
  cfg = open(args.config_file).read()
  eval(cfg, registry.get_symbols())


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Hyper parameter optimizer.')
  parser.add_argument('config_file', type=str, help='Configuration file.')
  args = parser.parse_args()
  main(args)
