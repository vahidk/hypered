"""Hyper parameter optimizer interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from interface import registry


OUTPUT_DIR = "experiments"


@registry.export
def set_output_dir(path):
  return OUTPUT_DIR


class experiment_dir(registry.exportable):

  def __call__(self, ctx):
    return ctx["experiment_dir"]


class params_path(registry.exportable):

  def __call__(self, ctx):
    return ctx["params_path"]


class results_path(registry.exportable):

  def __call__(self, ctx):
    return ctx["results_path"]


class device_id(registry.exportable):

  def __init__(self, count):
    self.count = count
    self.index = -1

  def __call__(self, ctx):
    self.index = (self.index + 1) % self.count
    return self.index
