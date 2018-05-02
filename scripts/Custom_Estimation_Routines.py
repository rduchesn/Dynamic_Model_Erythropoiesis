"""This library contains the functions necessary for our parameter estimations. They are based on a random sampling of the initial guess to assure convergence to the optimal parameter set.

The problem with this random sampling is that some initial guess will make the TNC algorithm crash, while others won't. This means that we have to sample initial guesses *until* we have found enough initial guesses that didn't make TNC crash. This is provided by the functions try_estimate, Estimate and Sample_Estimate. Sample_Estimate is the one that wraps up all the others, and it is thus the one that should be used.

This library also supports Latin Hypercube Sampling (LHS) of the initial guesses."""

import numpy as np
import scipy.optimize
import warnings
import CustomErrors
import scipy.stats
import Latin_Hypercube_Sampling as LHS

def try_estimate(func,n,args,init=None,full_output=False,verbose=True,bounds=(0,np.inf),maxeval=int(1e6)):
  """Basic estimator for the optimal parameters of a model, given data.

  Arguments:
  ----------
  func (function): objective function to be minimized. This is the function which outputs the -2logLikelihood of the model.
  n (int): number of parameters to be optimized
  args (tuple): additional arguments passed to func. Typically: data to be fitted, already estimated parameters, error model...
  bounds (tuple): bounds for the parameters values (tuple of length 2 tuples containing -np.inf, 0, or np.inf). Tells whether a parameter is strictly positive or can be negative
  init (array): initial guess for the parameters. Default to None (if random samples should be drawn, according to bounds)
  full_output (bool): whether the minimal error should be output together with the optimal parameter set or not
  verbose (bool): whether the information of the estimator should be displayed or not.
  maxeval: (integer) maximum number of iterations/function evaluations for estimation (we use the same for both arguments of scipy.optimize.minimize, because to be honest I can't really tell the difference between the two.)

  Returns:
  --------
  If full_output==True, the output is a dict with keys 'error' (value of the objective function at the optimal parameter set) and 'parameters' (optimal parameter values)
  Otherwise, the output is just the optimal parameter values."""

  if init is None:
    init=np.random.random(n)

  warnings.filterwarnings('error','',RuntimeWarning) #We turn all RuntimeWarnings into errors to avoid problems with certain parameters sets

  opt=scipy.optimize.minimize(func,init,args=args,method='TNC',bounds=bounds,options={'maxiter':maxeval})
  if opt.success:
    err=opt.fun
    if verbose:
      print('Estimation successful\nSum of squared errors:\t%.4g\nNumber of evaluations:\t%i\n'%(err,opt.nfev))
    if full_output:  #we output the optimal parameters and error if specified
      return({'error':err, 'parameters':opt.x})
    else:    #otherwise we only output the optimal parameters
      return(opt.x)

  else:  #if the estimation as failed...
    #if verbose_error:
    raise CustomErrors.EstimationError(opt.message)  #we raise an error with the information about the failure

def Estimate(func,n,args,bounds,init=None,full_output=False,verbose_success=True,verbose_error=True,maxeval=int(1e6),m=0):
  """Working estimator for the optimal parameter set of a model, given data. We say 'Working', because this won't bring in any error.
  For a detailed documentation of the options, please refer to the documentation of try_estimate in this library. There are three additional arguments to those of try_estimate:
  verbose_success (bool): whether or not the estimator should display a message when the estimation is successful (including the number of model evaluations and the value of the objective function).
  verbose_error (bool): whether or not the estimator should display a message when the estimation is not successful (including the error message).
  m (int): if Latin Hypercube Sampling is to be used, m is the sample size.

  Returns:
  --------
  opt: an output of try_estimate."""

  if type(init)==np.ndarray:
    if init.dtype==np.dtype('float64'):  #in this case the initial guess is given as argument to Estimate
      p0=init    #p0 is the initial guess
      opt=try_estimate(func,n,args,p0,full_output,verbose_success,bounds,maxeval)   #we estimate the optimal value for this set (we're not sure that the optimizer won't crash)
      return(opt)

  #If the initial guess is not given as argument, we have to randomly choose it. This random guess might make the estimator crash, so we need to randomly choose it *until* we find a value that can be optimized.
  success=False
  while not success:
    try:
      #we sample the initial guess
      if init is None:  #in this case we have to initiate a random value, according to the given bounds
        p0=10**(2*np.random.rand(n)-2)  #we initiate loguniform parameters between 0.1 and 10

      elif type(init)==np.ndarray:
        if init.dtype==np.dtype('int64'):  #in this case, it is the index of the LHS intervals which is given
          p0=10**(2*LHS.sample(n,m,init)-2)   #we initiate parameters between 0.1 and 10

      #now we can optimize
      opt=try_estimate(func,n,args,p0,full_output,verbose_success,bounds,maxeval)   #So we have an initial guess to optimize
      success=True                                                    #...and if we succeed we stop. ...
    except (CustomErrors.EstimationError,RuntimeWarning) as err:      #...Otherwise, if we don't succeed...
      if verbose_error:
        print('Estimation Error:',err)                                  #...we print what error happened and try again
  return(opt)

def Sample_Estimate(func,n,args,bounds,nsamples=500,lhs=True,maxeval=int(1e6),full_output=False,verbose_success=True,verbose_error=False):
  """Sampling-based estimator. Estimates the optimal parameters, based on a random sampling of the initial guess.

  Arguments:
  ----------
  For a detailed documentation of each argument, please refer to the documentation of functions Estimate and try_estimate in this library. Arguments unique to Sample_Estimate include:
  nsamples (int): sample size from which to draw the estimation
  lhs (boolean): whether or not the initial guesses should be drawn using Latin Hypercube Sampling (if False, parameters are log-uiformly drawn between 0.01 and 100)
  
  Returns:
  --------
  The optimal parameter set, in the same format as try_estimate."""

  #Initiating stuff
  Error=np.empty(nsamples)
  Parameters=np.empty((n,nsamples))
  if lhs:
    Intervals=LHS.Lhs(n,nsamples,'intervals') #intervals from which to draw each initial guess

  for i in range(nsamples):
    if verbose_success or verbose_error:
      print('Estimation %i/%i'%(i+1,nsamples))
    if lhs:
      opt=Estimate(func,n,args,bounds,init=Intervals[i],full_output=True,verbose_success=verbose_success,verbose_error=verbose_error,maxeval=maxeval,m=nsamples)
    else:
      opt=Estimate(func,n,args,bounds,init=None,full_output=True,verbose_success=verbose_success,verbose_error=verbose_error,maxeval=maxeval,m=nsamples)
    Error[i]=opt['error']
    Parameters[:,i]=opt['parameters']

  imin=np.argmin(Error)
  err=Error[imin]
  parameters=Parameters[:,imin]
  if verbose_success:
    print('Estimation process over.\nSum of squared errors:\t%.8g'%err)
  if full_output:
    return({'error':err,'parameters':parameters})
  else:
    return(parameters)
