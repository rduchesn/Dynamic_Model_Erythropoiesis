"""This library contains functions for sampling points in a hypercube using Latin Hypercube Sampling."""

import numpy as np

def sample(n,m,interval):
  """Sample one point in the n-dimensional unit hypercube.

  Arguments:
  ----------

  n: (int) dimension of the space;
  m: (int) number of segments along one dimension (i.e. there are m**n possible intervals from which to draw a sample)
  interval: (1*n array of ints between 0 and m-1) containing the coordinates of the block from which to draw the sample)

  Returns:
  --------
  s: (1*n array of floats in [0,1]): coordinates of the sampled point"""

  return((np.random.rand(n)+interval)/m)

def choose_interval(n,m,Remaining_Intervals):
  """Randomly chooses a block in the remaining blocks of the hypercube.

  Arguments:
  ----------
  n: dimension of the spae
  m: number of block along one dimension (there m**n blocks in total)
  Remaining_Intervals: list (of length n) of lists (of length k<m) containing the indeces of the unsampled blocks in each dimensions."""

  k=len(Remaining_Intervals[0])  #number of remaining blocks
  coords=np.random.randint(k,size=n)
  interval=list((Remaining_Intervals[i][coords[i]] for i in range(n))) #the index of the block along the i-th dimension is the i-th component of coords
  return(interval)

def Lhs(n,m,output):
  """Samples m points in the n-dimensional unit hypercube [0,1]**n

  Arguments:
  ----------
  n: (int) dimension of the space;
  m: (int) size of the desired sample;
  ouput: string specifying the desired output type:
    'values': outputs the array of the sampled points
    'intervals': outputs the array of the sampled intervals (required if you need several different samples at the same points)

  Returns:
  --------
  output: (m*n array) containing the requested output"""
   
  #initiating variables
  Remaining_Intervals=list(list(range(m)) for i in range(n)) #the remaining unsampled blocks list contains the indeces of the unexplored blocks alond each dimension
  if output=='values':
    Output=np.empty((m,n))
  elif output=='intervals':
    Output=np.empty((m,n),dtype=int)
  
  for i in range(m-1,-1,-1):   #we proceed by decreasing number of remaining blocks
    #first we pick the i-th point in the sample
    interval=choose_interval(n,m,Remaining_Intervals)
    if output=='values':
      s=sample(n,m,interval)
      Output[i]=s
    else:
      Output[i]=np.array([interval])

    #now we update the list of possible intervals
    for j in range(n):
      Remaining_Intervals[j].remove(interval[j])
  return(Output)
