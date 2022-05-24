"""Utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import hashlib
import json


def unwrap_dict(dic, flat=None, suffix=None, sep="."):
  if flat is None:
    flat = {}
  for k, v in dic.items():
    flat_suffix = suffix + sep + k if suffix else k
    if isinstance(v, dict):
      unwrap_dict(v, flat, flat_suffix)
    else:
      flat[flat_suffix] = v
  return flat


def wrap_dict(flat, dic=None, suffix=None, sep="."):
  if dic is None:
    dic = {}
  for k, v in flat.items():
    p = dic
    keys = k.split(sep)
    for k in keys[:-1]:
      if k not in p:
        p[k] = {}
      p = p[k]
    p[keys[-1]] = v
  return dic


def merge_dicts(base, *args):
  merged = base.copy()
  for arg in args:
    for k, v in arg.items():
      merged[k] = v
  return merged


def lookup_flat(dic, flat_key, sep="."):
  p = dic
  for k in flat_key.split(sep):
    p = p[k]
  return p


def deserialize_json(text):
  return ast.literal_eval(str(text))


def serialize_json(data, indent=4):
  return json.dumps(deserialize_json(data), indent=indent)


def hash_json(data):
  m = hashlib.md5(serialize_json(data, None).encode())
  return m.hexdigest()
