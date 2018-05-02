"""This script contains the functions necessary to simulate our SB model of in vitro differentiation."""

import numpy as np

def SB(t,init,rhoS,deltaSB,rhoB):
    """Simulates the model.

    Arguments:
    ----------
    t (nd.array): vector of simulation times
    init (length-2 list): initial condition
    rhoS, deltaSB, rhoB (floats): parameters of the model: self-renewing cells net growth rate, differentiation rate, differentiated cells growth rate.

    Return:
    -------
    SB (nd.array): values of the variable at the simulated timepoints."""

    #useful variables
    [S0,B0]=init
    lambdaS=deltaSB-rhoS

    #eigenvalues checking
    thresh=1e-5
    bSB=(-rhoB-lambdaS)*(abs(-rhoB-lambdaS)>=thresh)

    S=S0*np.exp(-lambdaS*t)

    #two cases for B:
    if bSB!=0:
        b2=deltaSB*S0/bSB; b1=B0-b2
        B=b1*np.exp(rhoB*t)+b2*np.exp(-lambdaS*t)

    else:
        b2=deltaSB*S0; b1=B0
        B=(b1+b2*t)*np.exp(-lambdaS*t)

    return(np.vstack((S,B)))

def TB(t, init, rhoS, deltaSB, rhoB):
    """Computes the observables of a simulation, with the same arguments as SB."""

    simul=SB(t, init, rhoS, deltaSB, rhoB)
    T=np.sum(simul, axis=0)
    TB=np.vstack((T,simul[1]))
    return(TB)
