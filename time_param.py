#!/usr/bin/env python3
'''
    This document defines all the time parameters.
    '''
import numpy as np

trials = 10                   # number of trials of the experiment
h = 0.01                        # [s] the sampling period
T = 20                          # [s] total time
N_t = int(round(T/h))           # [] total number of simulation steps
N = N_t+1
t_p = np.linspace(0,T,N)        # [s] time points
delta_N = 200                   # [] number of steps for time window
steps = range(0,N)              # [] range of simulation steps
