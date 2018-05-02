"""This file contains custom Error classes for the simulation and calibration of our deterministic models."""

class InputError(Exception):
  """Raised by improper inputs, such as wrong number of parameters in likelihood function."""
  def __init__(self, value):
    self.value = value

  def message(self):
    print('Invalid input: %s'%self.value)


class EstimationError(Exception):
  """Raised during an estimation with scipy.optimize.leastsq if something went wrong"""
  def __init__(self, value):
    self.value = value

class SimulationError(Exception):
  """Raised during a simulation by a RuntimeOverflow (exponential or product too large for numpy)"""

  def __init__(self, value):
    self.value=value

class ErrorError(Exception):
  """Though the name is funny, it is simply raised when an unknown error model is specified, either in a simulation or an estimation step."""
  def __init__(self, value):
    self.value=value

  def message(self):
    """Returns the error message that should be printed upon raising"""
    return('Unknown Error model specified: %s'%self.value)

class OutputError(Exception):
  """Raised during the computation of the Profile Likelihood wrt a parameter, of the requested output is unknown."""
  def __init__(self, value):
    """OutputType is the unknown output that was requested"""
    self.value = value

  def message(self):
    """Returns the error message that should be printed upon raising"""
    return('Unknown output requested: %s'%self.value)
