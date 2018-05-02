"""Contains the few functions necessary to implement error models in our simulations.

Three error models are available: Constant, Proportional and Combined. For each ErrorModel, the user can:
  - compute the -2logLikelihood of the model (with function logLikelihood_ErrorModel). This can then be minimized numerically to find the optimal parameter set describing a dataset.
  - evaluate the weighted residuals of the model (with function Residuals_ErrorModel). These should follow a normal distribution (with mean=0 and variance=1), which provides a visual test of the error model.

In order to use these functions, one needs two additional features:
  - a function which simulates the dynamical model
  - an experimental dataset to compare to the numerical simulation of the model. This can either be a single experiment, or a set of experiments to be fitted on average. The computations can be run on raw data, or on log-transformed data.

For the Constant Error model, there is a computational trick which allows to estimate the dynamical parameters and the error parameter separately (described in Appendix A of Lavielle & Bleakley, Mixed Effects Models for the Population Approach: Models, Tasks, Methods and Tools. Chapman & Hall/CRC Biostatistics Series. CRC Press/Taylor & Francis Group, 2014.)
"""

import numpy as np

def sum_of_squares(p, f, args):
  """Computes the sum of squares of the vector output of a function.
  p: input to the function
  f: function whose sum of squares at point p should be evaluated
  args: additional arguments.
  
  Returns:
  --------
  Sum of squares float."""
  
  return(np.sum(f(p, *args)**2))

def logLikelihood_ConstantError(Psi,Data,SimulModel,t,init,other_params=np.array([]),log=False,compute_alpha=True):
  """Computes the -log likelihood of a dynamic model, under the constant error model. It can use the trick that the error parameter optimum is set by the model residuals. However, using this trick is not mandatory.

  Arguments:
  ----------
  Psi (1D nd.array): Parameters of the model. If the error parameter alpha is provided, then Psi has the parameters of the dynamic model as first components and alpha is the last one. If alpha should be computed using residuals, then Psi only contains the parameters of the dynamic model.
  Data (ndarray): Experimental data to which the simulation should be compared.
  SimulModel (function): Function that simulates the model (it should output the simulation in the same matrix format as the Data array).
  t: vector of times for simulation
  init: initial state of the system under simulation.
  other_params: possible other parameters
  log (boolean): whether or not the error is to be computed on log-transformed data
  compute_alpha (boolean): whether the error parameter alpha is specified by the user or should be computed using the residuals of the model

  Returns:
  --------
  logL, which is the minus log likelihood of the model."""

  if not compute_alpha:
    alpha=Psi[-1]
    Theta=Psi[:-1]
  else:
    Theta=Psi
    
  if Theta.shape==(0,):
    DynParams=other_params
  else:
    if len(other_params.shape)>1:  #if the other parameters are not in the form of a vector
      other_params=np.concatenate(other_params)   #then we correct that
    DynParams=np.concatenate((other_params,Theta))

  Simulation=SimulModel(t,init,*DynParams)

  m = t.size      #number of timepoints
  if len(Simulation.shape) == 1:    #if there's one coordinate in the simulation
    n=1       #then there's just one variable
  else:
    n=Simulation.shape[0]  #otherwise there is one variable per coordinate in the simulation

  N = int(Data.size/(n*m))   #once we know the number of varaibles, and the number of timepoints, we have the number of experiments in the set.


  #if several datasets are used simultaneously, we have to correct the Simulation and the Errors.
  #if len(Data.shape)>0:

  Data=Data.reshape((N,n,m))
  Simulation=np.tile(Simulation, (N,1,1))

  if log:
    #if the data is to be log-transformed we threshold all vectors.
    thresh=np.finfo(float).eps
    Simulation=Simulation[np.where(Data>thresh)]
    data=Data[np.where(Data>thresh)]

    Residuals=np.log(data)-np.log(Simulation)
  else:
    data=Data
    Residuals=data-Simulation

  if compute_alpha:
    alpha=np.sqrt(np.mean(Residuals**2))
  Errors=alpha*np.ones(data.shape)

  Likelihood=(Residuals/Errors)**2+2*np.log(Errors)
  return(np.sum(Likelihood))

def Residuals_ConstantError(Psi,Data,SimulModel,t,init,other_params=np.array([]),log=False,compute_alpha=True):
  """Computes the weighted residuals of a dynamic model, under the constant error model. It can use the trick that the error parameter optimum is set by the model residuals, but using this trick can be mandatory.

  Arguments:
  ----------
  Psi (1D nd.array): Parameters of the model. If the error parameter alpha is provided, then Psi has the parameters of the dynamic model as first components and alpha is the last one. If alpha should be computed using residuals, then Psi only contains the parameters of the dynamic model.
  Data: Experimental data to which the simulation should be compared.
  SimulModel: Function that simulates the model
  t: vector of times for simulation
  init: initial state of the system under simulation.
  other_params: possible other parameters
  log (boolean): whether or not the error is to be computed on log-transformed data
  compute_alpha (boolean): whether the error parameter alpha is specified by the user or should be computed using the residuals of the model

  Returns:
  --------
  weighted_residuals (ndarray): weighted residuals of the model for each variable at each timepoint."""

  if not compute_alpha:
    alpha=Psi[-1]
    Theta=Psi[:-1]
  else:
    Theta=Psi
    
  if Theta.shape==(0,):
    DynParams=other_params
  else:
    if len(other_params.shape)>1:  #if the other parameters are not in the form of a vector
      other_params=np.concatenate(other_params)   #then we correct that
    DynParams=np.concatenate((other_params,Theta))

  Simulation=SimulModel(t,init,*DynParams)

  m = t.size      #number of timepoints
  if len(Simulation.shape) == 1:    #if there's one coordinate in the simulation
    n=1       #then there's just one variable
  else:
    n=Simulation.shape[0]  #otherwise there is one variable per coordinate in the simulation

  N = int(Data.size/(n*m))   #once we know the number of varaibles, and the number of timepoints, we have the number of experiments in the set.


  #if several datasets are used simultaneously, we have to correct the Simulation and the Errors.
  #if len(Data.shape)>0:

  Data=Data.reshape((N,n,m))
  Simulation=np.tile(Simulation, (N,1,1))

  if log:
    #if the data is to be log-transformed we threshold all vectors.
    thresh=np.finfo(float).eps
    Simulation=Simulation[np.where(Data>thresh)]
    data=Data[np.where(Data>thresh)]

    Residuals=np.log(data)-np.log(Simulation)
  else:
    data=Data
    Residuals=data-Simulation

  if compute_alpha:
    alpha=np.sqrt(np.mean(Residuals**2))
  Errors=alpha*np.ones(data.shape)
  
  weighted_residuals=Residuals/Errors
  return(weighted_residuals)

def logLikelihood_ProportionalError(Psi,Data,SimulModel,t,init,other_params=np.array([]),log=False):
  """Computes the -log likelihood of a dynamic model, under the proportional error model.
  Psi=array([Theta,beta]), where Theta are the parameters of the dynamic model, and beta is the error parameter.
  Data (ndarray): Experimental data to which the simulation should be compared. The first dimension separates the different datasets on which the likelihood evaluation is run (if there is only one dataset, this dimension is omitted). The second dimension separates variables. The third dimension separates timepoints.
  SimulModel: Function that simulates the model
  t: vector of times for simulation
  init: initial state of the system under simulation.
  other_params: possible other parameters
  log (boolean): whether or not the error is to be computed on log-transformed data
  Returns: logL, which is the minus log likelihood of the model."""

  beta=Psi[-1]
  Theta=Psi[:-1]
  if Theta.shape==(0,):
    DynParams=other_params
  else:
    DynParams=np.concatenate((other_params,Theta))
  
  Simulation=SimulModel(t,init,*DynParams)

  m = t.size      #number of timepoints
  if len(Simulation.shape) == 1:    #if there's one coordinate in the simulation
    n=1       #then there's just one variable
  else:
    n=Simulation.shape[0]  #otherwise there is one variable per coordinate in the simulation

  N = int(Data.size/(n*m))   #once we know the number of varaibles, and the number of timepoints, we have the number of experiments in the set.

  #if several datasets are used simultaneously, we have to correct the Simulation and the Errors.
  #if len(Data.shape)>0:

  Data=Data.reshape((N,n,m))
  Simulation=np.tile(Simulation, (N,1,1))

  Errors=beta*Simulation

  #we perform a thresholding on the error values to get rid of the null ones

  thresh=np.finfo(float).eps
  Errors=Errors[np.where(Data>thresh)]
  Simulation=Simulation[np.where(Data>thresh)]
  data=Data[np.where(Data>thresh)]

  if log:
    Errors=beta*np.log(data)
    Likelihood=((np.log(data)-np.log(Simulation))/Errors)**2+2*np.log(Errors)
  else:
    Likelihood=((data-Simulation)/Errors)**2+2*np.log(Errors)
  return(np.sum(Likelihood))

def Residuals_ProportionalError(Psi,Data,SimulModel,t,init,other_params=np.array([]),log=False):
  """Computes the weighted residuals of an estimated parameter set of a dynamic model, under the proportional error model.
  Psi=array([Theta,beta]), where Theta are the parameters of the dynamic model, and beta is the error parameter.
  Data: Experimental data to which the simulation should be compared.
  SimulModel: Function that simulates the model
  t: vector of times for simulation
  init: initial state of the system under simulation.
  other_params: possible other parameters
  log (boolean): whether or not the error is to be computed on log-transformed data
  Returns: Residuals, the flattened vector of weighted residuals."""

  beta=Psi[-1]
  Theta=Psi[:-1]
  if Theta.shape==(0,):
    DynParams=other_params
  else:
    DynParams=np.concatenate((other_params,Theta))
  
  Simulation=SimulModel(t,init,*DynParams)

  m = t.size      #number of timepoints
  if len(Simulation.shape) == 1:    #if there's one coordinate in the simulation
    n=1       #then there's just one variable
  else:
    n=Simulation.shape[0]  #otherwise there is one variable per coordinate in the simulation

  N = int(Data.size/(n*m))   #once we know the number of varaibles, and the number of timepoints, we have the number of experiments in the set.

  #if several datasets are used simultaneously, we have to correct the Simulation and the Errors.
  #if len(Data.shape)>0:

  Data=Data.reshape((N,n,m))
  Simulation=np.tile(Simulation, (N,1,1))

  Errors=beta*Simulation

  #we perform a thresholding on the error values to get rid of the null ones

  thresh=np.finfo(float).eps
  Errors=Errors[np.where(Data>thresh)]
  Simulation=Simulation[np.where(Data>thresh)]
  data=Data[np.where(Data>thresh)]

  if log:
    Errors=beta*np.log(data)
    Residuals=(np.log(data)-np.log(Simulation))/Errors
  else:
    Residuals=(data-Simulation)/Errors
  return(Residuals)

def logLikelihood_CombinedError(Psi,Data,SimulModel,t,init,other_params=np.array([]),log=False):
  """Computes the vector of loglikelihoods of a dynamic model using a combined error model.
  Psi=array([Theta,alpha,beta]), where Theta are the parameters of the dynamic model, and alpha and beta are the error parameters.
  Data: data compared to simulations.
  SimulModel: Function that simulates the model
  t: vector of the times for simulation
  init: initial guess for the parameters
  log (boolean): whether or not the error is to be computed on log-transformed data
  Returns: a vector whose sum of squares is the -logLikelihood of the model."""

  alpha=Psi[-2]
  beta=Psi[-1]
  Theta=Psi[:-2]
  if Theta.shape==(0,):
    DynParams=other_params
  else:
    DynParams=np.concatenate((other_params,Theta))

  Simulation=SimulModel(t,init,*DynParams)

  m = t.size      #number of timepoints
  if len(Simulation.shape) == 1:    #if there's one coordinate in the simulation
    n=1       #then there's just one variable
  else:
    n=Simulation.shape[0]  #otherwise there is one variable per coordinate in the simulation

  N = int(Data.size/(n*m))   #once we know the number of varaibles, and the number of timepoints, we have the number of experiments in the set.

  #if several datasets are used simultaneously, we have to correct the Simulation and the Errors.
  #if len(Data.shape)>0:

  Data=Data.reshape((N,n,m))
  Simulation=np.tile(Simulation, (N,1,1))

  #we perform a thresholding on the error values to get rid of the null ones
  #(apparently this wasn't done automatically, for a reason I can't understand)

  thresh=np.finfo(float).eps
  Simulation=Simulation[np.where(Data>thresh)]
  data=Data[np.where(Data>thresh)]

  #print(Simulation)
  #print(Errors)

  if log:
    Errors=alpha+beta*np.log(Simulation)
    Likelihood=((np.log(data)-np.log(Simulation))/Errors)**2+2*np.log(Errors)
  else:
    Errors=alpha+beta*Simulation
    Likelihood=((data-Simulation)/Errors)**2+2*np.log(Errors)
  return(np.sum(Likelihood))

def Residuals_CombinedError(Psi,Data,SimulModel,t,init,other_params=np.array([]),log=False):
  """Computes the vector of weighted residuals of a dynamic model using a combined error model.
  Psi=array([Theta,alpha,beta]), where Theta are the parameters of the dynamic model, and alpha and beta are the error parameters.
  Data: data compared to simulations.
  SimulModel: Function that simulates the model
  t: vector of the times for simulation
  init: initial guess for the parameters
  log (boolean): whether or not the error is to be computed on log-transformed data
  Returns: Residuals, the vector of weighted residuals."""

  alpha=Psi[-2]
  beta=Psi[-1]
  Theta=Psi[:-2]
  if Theta.shape==(0,):
    DynParams=other_params
  else:
    DynParams=np.concatenate((other_params,Theta))

  Simulation=SimulModel(t,init,*DynParams)
  m = t.size      #number of timepoints
  if len(Simulation.shape) == 1:    #if there's one coordinate in the simulation
    n=1       #then there's just one variable
  else:
    n=Simulation.shape[0]  #otherwise there is one variable per coordinate in the simulation

  N = int(Data.size/(n*m))   #once we know the number of varaibles, and the number of timepoints, we have the number of experiments in the set.

  #if several datasets are used simultaneously, we have to correct the Simulation and the Errors.
  #if len(Data.shape)>0:

  Data=Data.reshape((N,n,m))
  Simulation=np.tile(Simulation, (N,1,1))

 
  #we perform a thresholding on the error values to get rid of the null ones
  #(apparently this wasn't done automatically, for a reason I can't understand)

  thresh=np.finfo(float).eps
  Simulation=Simulation[np.where(Data>thresh)]
  data=Data[np.where(Data>thresh)]

  if log:
    Errors=alpha+beta*np.log(Simulation)
    Residuals=(np.log(data)-np.log(Simulation))/Errors
  else:
    Errors=alpha+beta*Simulation
    Residuals=(data-Simulation)/Errors
  return(Residuals)
