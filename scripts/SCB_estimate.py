"""This script contains functions for the simulation of our SCB model of in vitro erythroid differentiation."""

import numpy as np

def S(t,S0,rhoS):
  """Simulates the dynamics of self-renewing cells in a self-renewing (LM1) medium
  t (ndarray): vector of simulation times
  S0 (float): initial number of cells
  rhoS (float): net self-renewal rate of the cells."""

  return(S0*np.exp(rhoS*t))

def SCB(t,init,rhoS,deltaSC,rhoC,deltaCB,rhoB):
  """Simulates the SCB model.
  t (ndarray): vector of simulation times
  init=[S0,C0,B0]: initial condition
  rhoS, deltaSC, rhoC, deltaCB, rhoB (floats): parameters of the system."""

  #Declaring useful parameters
  [S0,C0,B0]=init
  lambdaS=deltaSC-rhoS  #Introducing global decay rates lambda.
  lambdaC=deltaCB-rhoC
  lambdaB=-rhoB

  #Checking for eigenvalues equalitites
  thresh=1e-5 #threshold difference for considering two eignevalues as equal
  bSC=(lambdaC-lambdaS)*(abs(lambdaC-lambdaS)>=thresh)
  bSB=(lambdaB-lambdaS)*(abs(lambdaB-lambdaS)>=thresh)
  bCB=(lambdaB-lambdaC)*(abs(lambdaB-lambdaC)>=thresh)

  #S has always the same expression
  S=S0*np.exp(-lambdaS*t)

  #there are two cases for C
  if bSC!=0:
    c2=deltaSC*S0/bSC; c1=C0-c2
    C=c1*np.exp(-lambdaC*t)+c2*np.exp(-lambdaS*t)

    #there are three subcases for B in that case
    if bCB==0:
      b2=deltaCB*c1; b3=deltaCB*c2/bSB; b1=B0-b3
      B=(b1+b2*t)*np.exp(-lambdaB*t)+b3*np.exp(-lambdaS*t)

    elif bSB==0:
      b2=deltaCB*c1/bCB; b3=deltaCB*c2; b1=B0-b2
      B=(b1+b3*t)*np.exp(-lambdaB*t)+b2*np.exp(-lambdaC*t)

    else:
      b2=deltaCB*c1/bCB; b3=deltaCB*c2/bSB; b1=B0-b2-b3
      B=b1*np.exp(-lambdaB*t)+b2*np.exp(-lambdaC*t)+b3*np.exp(-lambdaS*t)

  else:
    c2=deltaSC*S0
    c1=C0
    C=(c1+c2*t)*np.exp(-lambdaS*t)

    #there are two subcases for B in that case
    if bCB!=0:
      b3=deltaCB*c2/bSB; b2=(deltaCB*c1-b3)/bSB; b1=B0-b2
      B=b1*np.exp(-lambdaB*t)+(b2+b3*t)*np.exp(-lambdaC*t)

    else:
      b1=B0; b2=deltaCB*c1; b3=deltaCB*c2/2
      B=(b1+b2*t+b3*t**2)*np.exp(-lambdaB*t)

  return(np.vstack((S,C,B)))

def TB(t,init,rhoS,deltaSC,rhoC,deltaCB,rhoB):
  """Simulates the SCB equations, and extracts the observables of the system.
  Arguments:
  ----------
  same as for the SCB function.

  Returns:
  --------
  Y (ndarray): Matrix containing T=S+C+B as a function of time on the first row and B on the second row."""

  y=SCB(t,init,rhoS,deltaSC,rhoC,deltaCB,rhoB)
  T=np.sum(y,axis=0)
  Y=np.vstack((T,y[2]))
  return(Y)
