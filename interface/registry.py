"""Register interface symbols."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six


_EXPORTED = {}


def get_symbols():
  return _EXPORTED


def export(symbol):
  _EXPORTED[symbol.__name__] = symbol
  return symbol


class exportable_meta(type):
  def __new__(cls, clsname, parents, attrs):
    newclass = super(exportable_meta, cls).__new__(
      cls, clsname, parents, attrs)
    _EXPORTED[clsname] = newclass
    return newclass


@six.add_metaclass(exportable_meta)
class exportable(object):
  pass
