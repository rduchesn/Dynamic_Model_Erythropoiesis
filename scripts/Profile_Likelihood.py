"""Profile Likelihood computation for Dynamic Models
-------------------------------------------------
This library contains the functions needed for computing the Profile Likelihood of Dynamic Models.
It can be used to assess the Identifiability of such models, as described in Raue et al., Bioinformatics, 2009.
The algorithms that are used herein are from this paper, and can also be found on the webpage of Dr. Andreas Raue (http://www.fdmold.uni-freiburg.de/~araue/pmwiki.php/Projects/ProfileLikelihoodApproach)

The Library also contains functions that use the result of Profile Likelihood Computation, for example to quickly estimate the borders of the parameters confidence intervals, or to plot stuff."""

import numpy as np
import scipy
from scipy.stats import *
import matplotlib.pyplot as plt

import Custom_Estimation_Routines as CER

def init_step_size(parameters,parameter_index,bounds,likelihood_function,likelihood_args,d_par_init=0.1,d_likelihood=0.1,max_step=3,alpha=0.95):
  """Use this function to determine the step size between two successive points in a profile.
  Starting from the last estimate of the profile parameters, it determines the optimal step size between the last estimate and the next one. The step is null for all parameters except the one whose profile is being computed, and is optimal when the relative likelihood increase it induces is closest to a target value.

  Arguments:
  ----------
  parameters (1D np.array): parameters estimated at the last point of the profile
  parameter_index (int): index of the parameter whose profile likelihood is being computed
  bounds (tuple of length 2): the boundaries of the parameter of interest
  likelihood_function (function): function for computing the likelihood of the model
  likelihood_args (tuple): additional arguments for the likelihood function (typically: data, simulation function, time points vector...)
  d_par_init (float): initial guess for the step size
  d_likelihood (float): targeted relative likelihood variation between two points
  max_step(int): maximum number of step size evaluations before considering the profile as flat
  alpha (float): level of confidence for the parameter of interest

  Returns:
  --------
  d_par (float): optimal step size"""

  likelihood = likelihood_function(parameters, *likelihood_args)
  df = parameters.shape[0] #number of parameters = number of degrees of freedom
  chi2_threshold = scipy.stats.chi2.ppf(alpha,df) #likelihood-threshold of the confidence interval

  #initial guess for the step
  param_tmp = np.copy(parameters)
  d_par=d_par_init
  param_tmp[parameter_index] = parameters[parameter_index] + d_par

  #now we correct the initial guess if it is out of bonds.
  lower_bound , upper_bound = bounds
  if lower_bound==None:
    lower_bound=-np.inf
  if upper_bound==None:
    upper_bound=np.inf
  while param_tmp[parameter_index] > upper_bound or param_tmp[parameter_index] < lower_bound:  #if the current step jumps out of the parameter's bounds, then we reduce it
    print("Boundary reached")
    d_par /= 2
    param_tmp[parameter_index] = parameters[parameter_index] + d_par
    print('New value: %.4g'%param_tmp[parameter_index])
      
  d_chi2 = likelihood_function(param_tmp, *likelihood_args) - likelihood

  step_evaluations = 0 #number of evaluations of the step size
  #if the step is too big we reduce it
  if d_chi2 > chi2_threshold*d_likelihood:
    while d_chi2 > chi2_threshold*d_likelihood and step_evaluations < max_step and param_tmp[parameter_index] > lower_bound and param_tmp[parameter_index] < upper_bound:
      d_par /= 2
      param_tmp[parameter_index] = parameters[parameter_index] + d_par
      d_chi2 = likelihood_function(param_tmp, *likelihood_args) - likelihood
      step_evaluations += 1

  #otherwise we increase it
  else:
    while d_chi2 < chi2_threshold*d_likelihood and step_evaluations < max_step and param_tmp[parameter_index] > lower_bound and param_tmp[parameter_index] < upper_bound:
      d_par *= 2
      param_tmp[parameter_index] = parameters[parameter_index] + d_par
      d_chi2 = likelihood_function(param_tmp, *likelihood_args) - likelihood
      step_evaluations += 1
    d_par /= 2 #this is in Raue's algorithm but I don't really get it. Apparently the last doubling step is too much.

  return(d_par)

def profile_likelihood(parameters,parameter_index,parameter_value,likelihood_function,likelihood_args):
  """This function computes the likelihood of a model with a fixed value for one of its parameters (which is necessary for computing its Profile Likelihood).

  Arguments:
  ----------
  parameters (1D nd.array): If the model has n parameters, this is a length n-1 vector containing the values of the free parameters (i.e. those that are not fixed)
  parameter_index (int): index of the fixed parameter
  parameter_value (float): value of the fixed parameter
  likelihood_function (function): likelihood function of the model
  likelihood_args (tuple): additional arguments to the likelihood function (e.g. data, simulation function...)

  Returns:
  --------
  L: an output of likelihood_function (typically, a float)"""

  if isinstance(parameter_value, float):
    parameter=np.array([parameter_value])
  else:
    parameter=parameter_value
  full_parameters=np.insert(parameters,parameter_index,parameter)
  pl = likelihood_function(full_parameters, *likelihood_args)
  return(pl)

def Compute_Profile(parameters,parameter_index,likelihood_function,likelihood_args,bounds,target_sample_size=100,max_sample_size=1000,d_par_init=0.002,max_step=10,number_initial_guess_samples=30,alpha=0.95,verbose_success=False,verbose_error=False):
  """This function computes the Profile Likelihood of a dynamic model with respect to one of its parameters. Starting at the best-fit parameter set, it tries to increase the -log Likelihood of the model up to the identifiability threshold on each side of the optimum, up to a certain extent. If it doesn't reach the identifiability threshold during the authorized number of step size evaluations, it considers the parameter profile likelihood as unbounded on that side, and stops computing it.

  Arguments:
  ----------
  parameters (1D np.array): parameters estimated at the last point of the profile
  parameter_index (int): index of the parameter whose profile likelihood is being computed
  likelihood_function (function): function for computing the likelihood of the model
  likelihood_args (tuple): additional arguments for the likelihood function (typically: data, simulation function, time points vector...)
  bounds (tuple): bounds for the estimates of each estimated parameter (tuple of length-2 tuples). This means that the fixed parameter should not be mentioned.
  target_sample_size (int): expected number of points on each side of the optimum
  max_sample_size (int): maximum tolerated number of points on each side of the optimum
  d_par_init (float): initial guess for the step size
  max_step(int): maximum number of step size evaluations
  number_initial_guess_samples (int): number of independent runs of the parameter estimation for each point of the profile
  alpha (float): level of confidence for the parameter of interest

  Returns:
  --------
  Output (dict): Output containing the following entries:
    'Parameters' (n*(2*sample_size+1) array): Values of the parameters along the profile
    'Profile_Likelihood' (1D array): Values of the Profile Likelihood along the profile."""

  chi2 = likelihood_function(parameters, *likelihood_args)
  df = parameters.shape[0]  # number of parameters of the model
  chi2_threshold = scipy.stats.chi2.ppf(alpha,df) #likelihood-threshold of the confidence interval

  #we store the coordinates of the optimum
  params_backup = np.copy(parameters)
  chi2_backup = chi2

  #we intialize the output, and start filling it out
  Chi2PL=np.array([chi2])
  Parameters=np.transpose(np.array([parameters]))

  d_likelihood = 1/target_sample_size #the number of steps should be the inverse of the stepwise relative likelihood increase (see Supp. Inf. of raue et al., Bioinfo., 2009 for more detail)

  #For decreasing values of the parameter:
  params = np.copy(parameters)
  i=0
  #for i in range(sample_size):
  while i<max_sample_size and chi2-chi2_backup < 1.1*chi2_threshold:
    print("Computing point #%i of the profile"%i)
    d_par=init_step_size(params, parameter_index, bounds[parameter_index], likelihood_function, likelihood_args, - d_par_init*np.abs(parameters[parameter_index]), d_likelihood, max_step, alpha)
    params[parameter_index] += d_par

    opt=CER.Sample_Estimate(profile_likelihood, df-1, args=(parameter_index, params[parameter_index], likelihood_function, likelihood_args), bounds = bounds[:parameter_index]+bounds[(parameter_index+1):], nsamples = number_initial_guess_samples, full_output = True, verbose_success = verbose_success, verbose_error=verbose_error, lhs=False)

    #We update stuff
    params=np.insert(opt['parameters'],parameter_index,params[parameter_index])
    Parameters=np.insert(Parameters,0,params,axis=1)
    chi2=opt['error']
    Chi2PL = np.insert(Chi2PL, 0, chi2)
    i+=1

  #Resetting the original values of stuff
  params = np.copy(params_backup)
  chi2 = chi2_backup

  #For increasing values of the parameter:
  i=0
  while i<max_sample_size and chi2 - chi2_backup < 1.1*chi2_threshold:
    print("Computing point #%i of the profile"%i)
    d_par=init_step_size(params, parameter_index, bounds[parameter_index], likelihood_function, likelihood_args, d_par_init*np.abs(parameters[parameter_index]), d_likelihood, max_step, alpha)
    params[parameter_index] += d_par

    opt=CER.Sample_Estimate(profile_likelihood, df-1, args=(parameter_index, params[parameter_index], likelihood_function, likelihood_args), bounds = bounds[:parameter_index]+bounds[(parameter_index+1):], nsamples = number_initial_guess_samples, full_output = True, verbose_success = verbose_success, verbose_error = verbose_error, lhs=False)

    #We update stuff
    params=np.insert(opt['parameters'],parameter_index,params[parameter_index])
    Parameters=np.append(Parameters,np.transpose(np.array([params])),axis=1)
    chi2 = opt['error']
    Chi2PL = np.append(Chi2PL, chi2)
    i+=1

  return({'Parameters': Parameters, 'Profile_Likelihood':Chi2PL})
    
def Confidence_Interval(Profile, parameter_index, alpha=0.95):
  """Given the Profile of a Likelihood function with respect to one of its parameters, this function extracts the borders of its confidence interval at a certain confidence level alpha.

  Arguments:
  ----------
  Profile (dict): Profile dictionary, typically the output of the Compute_Profile Function from this library. It contains one 'Parameters' keys containing the values of the parameters at each point of the profile, and one 'Profile_Likelihood' key, which contains the values of the PL.
  parameter_index (int): the index of the parameter whose CI are to be computed
  alpha (float, between 0 and 1): required level of confidence.

  Returns:
  --------
  (theta_min, theta_max): tuple containing the borders of the confidence interval, or None if the CI is unbounded on one direction."""

  df, number_points=Profile['Parameters'].shape   #number of parameters of the model and number of points in the profile
  opt_likelihood=np.min(Profile['Profile_Likelihood'])
  opt_index=np.argmin(Profile['Profile_Likelihood']) #first index of an optimum

  threshold=opt_likelihood + scipy.stats.chi2.ppf(alpha, df)  #threshold for identifiability
  distance_to_threshold=(Profile['Profile_Likelihood']-threshold)**2

  lower_index=np.argmin(distance_to_threshold[:(opt_index + (opt_index==0))]) #index of the lower bound (we shift the limit by 1 in case the optimum is on the border of the profile)
  upper_index=np.argmin(distance_to_threshold[(opt_index - (opt_index==number_points)):])+opt_index #index of the upper bound

  return((Profile['Parameters'][parameter_index,lower_index], Profile['Parameters'][parameter_index,upper_index]))

def Read_Profile(input_file):
  """Reads the profile stored in a file into a friendly format"""

  Data=np.genfromtxt(input_file)
  Data={'Parameters':Data[:-1], 'Profile_Likelihood':Data[-1]}
  return(Data)

def Plot_Profile(Profile,Parameter_index,alpha=0.95,show=True,output_file=None,xtitle='',ytitle='',maintitle=''):
  """Plots the profile of a parameter.
  Arguments:
  ----------
  Profile (dict): profile of the parameter. It has two keys, 'Parameters' is a (Nxm) array (N is the number of parameters, m the number of points) containing the parameter values along the profile, and 'Profile_Likelihood' is a (m,) array containing the likelihood values.
  Parameter_index (int): index of the paramter of interest.
  alpha (float): requested level of confidence.
  show (bool): should the figure be displayed?
  ouput_file (str or None): path to the file where the graphic should be output.
  xtitle (str): label for x axis
  ytitle (str): label for y axis
  title (str): general title for the figure"""


  plt.clf()
  df=Profile['Parameters'].shape[0]  #number of estimated parameters
  threshold=np.min(Profile['Profile_Likelihood']) + chi2.ppf(alpha,df)
  plt.plot(Profile['Parameters'][Parameter_index], Profile['Profile_Likelihood'], '.', c='0.2', linewidth=2)
  plt.plot([Profile['Parameters'][Parameter_index, 0], Profile['Parameters'][Parameter_index, -1]], [threshold, threshold], '--', c='0.2', linewidth=2)
  plt.xlabel(xtitle,fontsize=12)
  plt.ylabel(ytitle,fontsize=12)
  plt.title(maintitle,fontsize=12)

  if output_file!=None:
    plt.rcParams['figure.figsize']=5,5
    plt.savefig(output_file,dpi='figure',bbox_inches='tight')
  if show:
    plt.show()

def Plot_Two_Profiles(Profile1,Profile2,Parameter_index,alpha=0.95,show=True,output_file=None,xtitle='',ytitle='',label1='',label2='',maintitle=''):
  """Plots the comparison of two profile likelihood curves for the same parameter
  Arguments:
  ----------
  Profile1 (dict): first profile dict
  Profile2 (dict): second profile dict
  Parameter_index (int): index of the paramter of interest.
  label1 (str): label for the first profile curve
  label2 (str): label for the second profile curve
  For more doc on the other parameters, please refer to the docstring of the plot_profile function, in this same library."""

  df=Profile1['Parameters'].shape[0]  #number of estimated parameters

  threshold1=np.min(Profile1['Profile_Likelihood']) + chi2.ppf(alpha,df)
  threshold2=np.min(Profile2['Profile_Likelihood']) + chi2.ppf(alpha,df)

  plt.clf()
  plt.plot(Profile1['Parameters'][Parameter_index], Profile1['Profile_Likelihood'], '-', c='0.2', linewidth=2, label=label1)
  plt.plot(Profile2['Parameters'][Parameter_index], Profile2['Profile_Likelihood'], '-', c='#b50303', linewidth=2, label=label2)
  plt.plot([Profile1['Parameters'][Parameter_index, 0], Profile1['Parameters'][Parameter_index, -1]], [threshold1, threshold1], '--', c='0.2', linewidth=2)
  plt.plot([Profile2['Parameters'][Parameter_index, 0], Profile2['Parameters'][Parameter_index, -1]], [threshold2, threshold2], '--', c='#b50303', linewidth=2)
  plt.xlabel(xtitle,fontsize=12)
  plt.ylabel(ytitle,fontsize=12)
  plt.title(maintitle,fontsize=12)
  plt.legend(loc='best',fontsize=12)

  if output_file!=None:
    plt.rcParams['figure.figsize']=5,5
    plt.savefig(output_file,dpi='figure',bbox_inches='tight')
  if show:
    plt.show()
