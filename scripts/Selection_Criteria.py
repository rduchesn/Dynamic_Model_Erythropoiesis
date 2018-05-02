"""This library contains a few dummy functions for computing selection criteria on our models."""

import numpy as np

def AIC(L,k):
  """Computes the Akaike's Information Criterion of a model.
  L (float): -2 log(Likelihood) of the model (log in base e)
  k (int)  : number of parameters"""

  return(L+2*k)

def AICc(L,n,k):
  """Computes the AIC of a model, corrected for sample finiteness.
  L (float): -2 log(Likelihood) of the model (log in base e)
  k (int)  : number of model parameters
  n (int)  : sample size (number of points in the dataset)"""

  return(AIC(L,k)+2*k*(k+1)/(n-k-1))
