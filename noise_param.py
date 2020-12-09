#!/usr/bin/env python3
'''
    This document defines all the noise parameters.
    '''
import numpy as np

# if colored noise is chosen
s = 0.25                        # correlation kernel variance
gamma = 1/s**2                  # roughness parameter
varw = np.array([0.01, 0.01])   # uncertainty in states (variance)
varz = np.array([0.01])         # uncertainty in outputs (variance)
varw_noise = np.array([0.01, 1])   # noisy situation

# if gray noise is chosen
std = 0.5                       # standard deviation of the noise
