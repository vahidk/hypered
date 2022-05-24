"""Hyper parameter optimizer interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import skopt

from interface import registry


@registry.export
def uniform():
  return "uniform"


@registry.export
def log_uniform():
  return "log-uniform"


class variable(registry.exportable):
  pass


class real(variable):

  def __init__(self, low, high, prior=uniform):
    self.low = low
    self.high = high
    self.prior = prior

  def __call__(self):
    var = skopt.space.Real(self.low, self.high, self.prior())
    return var


class integer(variable):

  def __init__(self, low, high):
    self.low = low
    self.high = high

  def __call__(self):
    var = skopt.space.Integer(self.low, self.high)
    return var


class categorical(variable):

  def __init__(self, categories):
    self.categories = categories

  def __call__(self):
    var = skopt.space.Categorical(self.categories)
    return var
