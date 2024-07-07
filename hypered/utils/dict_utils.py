"""Utility functions for dictionary manipulation, JSON serialization, and hashing."""

from typing import Any, Optional, Callable

import ast
import hashlib
import json


def unwrap_dict(
    dic: dict,
    flat: Optional[dict] = None,
    suffix: Optional[str] = None,
    sep: str = ".",
) -> dict:
    """
    Recursively flattens a nested dictionary.

    Args:
        dic (dict): The dictionary to flatten.
        flat (dict, optional): The dictionary to store the flattened key-value pairs. Defaults to None.
        suffix (str, optional): The suffix for the current level of the dictionary. Defaults to None.
        sep (str, optional): The separator to use between nested keys. Defaults to ".".

    Returns:
        dict: The flattened dictionary.
    """
    if flat is None:
        flat = {}
    for k, v in dic.items():
        flat_suffix = suffix + sep + k if suffix else k
        if isinstance(v, dict):
            unwrap_dict(v, flat, flat_suffix)
        else:
            flat[flat_suffix] = v
    return flat


def wrap_dict(
    flat: dict,
    dic: Optional[dict] = None,
    suffix: Optional[str] = None,
    sep: str = ".",
) -> dict:
    """
    Converts a flattened dictionary back to a nested dictionary.

    Args:
        flat (dict): The flattened dictionary to convert.
        dic (dict, optional): The dictionary to store the nested key-value pairs. Defaults to None.
        suffix (str, optional): The suffix for the current level of the dictionary. Defaults to None.
        sep (str, optional): The separator used between nested keys in the flattened dictionary. Defaults to ".".

    Returns:
        dict: The nested dictionary.
    """
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


def merge_dicts(base: dict, *args) -> dict:
    """
    Merges multiple dictionaries into a base dictionary.

    Args:
        base (dict): The base dictionary to merge into.
        *args (dict): Additional dictionaries to merge.

    Returns:
        dict: The merged dictionary.
    """
    merged = base.copy()
    for arg in args:
        for k, v in arg.items():
            merged[k] = v
    return merged


def join_dicts(objs: list[Any], fn: Callable = list):
    """
    Joins multiple dictionaries together.

    Args:
        objs (list[Any]): The list of dictionaries to join
        fn (Callable, optional): The function to use for joining the values. Defaults to list.

    Returns:
        dict: The merged dictionary.
    """
    if not objs:
        return {}
    if isinstance(objs[0], dict):
        return {k: join_dicts([obj[k] for obj in objs], fn=fn) for k in objs[0].keys()}
    else:
        return fn(objs)


def lookup_flat(dic: dict, flat_key: str, sep: str = ".") -> Any:
    """
    Looks up a value in a nested dictionary using a flattened key.

    Args:
        dic (dict): The dictionary to look up the value in.
        flat_key (str): The flattened key representing the path to the value.
        sep (str, optional): The separator used between nested keys in the flattened key. Defaults to ".".

    Returns:
        Any: The value found at the specified key path.
    """
    p = dic
    for k in flat_key.split(sep):
        p = p[k]
    return p


def deserialize_json(text: str) -> Any:
    """
    Deserializes a JSON string into a Python object.

    Args:
        text (str): The JSON string to deserialize.

    Returns:
        Any: The deserialized Python object.
    """
    return ast.literal_eval(str(text))


def serialize_json(data: Any, indent: Optional[int] = 4) -> str:
    """
    Serializes a Python object into a JSON string with indentation.

    Args:
        data (Any): The Python object to serialize.
        indent (int, optional): The number of spaces to use for indentation. Defaults to 4.

    Returns:
        str: The serialized JSON string.
    """
    return json.dumps(deserialize_json(data), indent=indent)


def hash_json(data: Any) -> str:
    """
    Computes the MD5 hash of a serialized JSON object.

    Args:
        data (Any): The Python object to hash.

    Returns:
        str: The MD5 hash of the serialized JSON object.
    """
    m = hashlib.md5(serialize_json(data, None).encode())
    return m.hexdigest()
