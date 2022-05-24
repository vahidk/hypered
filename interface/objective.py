"""Hyper parameter optimizer interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from common import utils
from interface import registry


@registry.export
def minimize(name):
  def _func(metrics):
    return utils.lookup_flat(metrics, name)
  return _func


@registry.export
def maximize(name):
  def _func(metrics):
    return -utils.lookup_flat(metrics, name)
  return _func
