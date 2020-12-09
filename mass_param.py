#!/usr/bin/env python3
'''
    This document defines all the parameters for each mass that is attached to a spring-damper system.
    '''
import numpy as np
from scipy.linalg import toeplitz

class Mass_Spring_Damper:
    """ Specify the parameters of a mass-spring-damper system in state space form, i.e. dx = Ax + Bu + w and y = Cx + z.
        Input parameters:
        m - mass in kg
        k - spring constant in N/m
        c - damping constant in Ns/m
        """
    
    def __init__(self, mass, spring, damper):
        self.m = mass
        self.k = spring
        self.c = damper
        self.A = np.array([[0, 1], [-self.k/self.m, -self.c/self.m]])
        self.B = np.array([[0], [1/self.m]])
        self.C = np.array([[1, 0]])
    
    def input(self, input_sequence,input_sequence_name):
        self.u = np.vstack(input_sequence)
        self.u_name = input_sequence_name
    
    def make_colored_noise(self,varz,varw,s,t_p,trials):
        n = self.A.shape[1]
        q = self.C.shape[0]
        N = t_p.size
        # Temporal convolution matrix:
        T = toeplitz(np.exp(-t_p**2/(2*s**2)))
        K = np.diag(1./np.sqrt(np.diag(T.dot(T.conj().T))))*T
        
        # Generate state and measurement noise:
        Pw = np.diag(1./varw)
        Pz = np.diag(1./varz)
        '''
            self.w = np.sqrt(np.linalg.inv(Pw)).dot(np.random.randn(n,N)).dot(K)
            self.z = np.sqrt(np.linalg.inv(Pz)).dot(np.random.randn(q,N)).dot(K)
            '''
        w = np.zeros((n,N,trials))
        z = np.zeros((q,N,trials))
        for i in range(trials):
            w[:,:,i] = np.sqrt(np.linalg.inv(Pw)).dot(np.random.randn(n,N)).dot(K)
            z[:,:,i] = np.sqrt(np.linalg.inv(Pz)).dot(np.random.randn(q,N)).dot(K)
        self.w = w
        self.z = z

'''
    def make_random_noise(self,std,N_t):
    self.w = np.random.normal(0,std,(2,N))    # random noise with 0 mean and x std
    '''

'''Time parameters
    trials      - number of trials of the experiment
    h           - [s] the sampling period
    T           - [s] total time
    N           - [] total number of simulation steps
    t_p         - [s] time points
    delta_N     - [] number of steps for time window
    steps       - [] range of simulation steps
    '''
from time_param import trials,h,T,N,t_p,delta_N,steps
'''Noise parameters
    varz        - variance in states
    varw        - variance in outputs
    varw_noise  - noisy variance in states
    s           - correlation kernel variance
    std         - gray noise standard deviation
    '''
from noise_param import varz,varw,varw_noise,s

#--initialize the mass-spring-damper systems
x0 = np.random.rand(trials,2)-0.5
n_m = 3

print("Choose the parameters of the state-space, where experiment 1 makes a data that should lead to correct conclusions and experiment 2 makes a data that should lead to incorrect conclusions.")
while True:
    try:
        experiment_number = input("Enter the number of the experiment (1A, 1B, 1C, 2A, 2B or 2C) and press enter:")
        # EXPERIMENT 1: PARAMETERS FOR CORRECT CONCLUSION
        if experiment_number=='1A':
            # experiment 1A
            mass1 = Mass_Spring_Damper(2.5, 6, 2)
            mass2 = Mass_Spring_Damper(2.5, 6, 2)
            mass3 = Mass_Spring_Damper(2, 4, 4)
            mass1.input(2*np.cos(4*t_p)+3,'$2cos(4t_p)+3$')
            mass2.input(np.zeros(N),'0')
            mass3.input(np.zeros(N),'0')
            # set noise
            mass1.make_colored_noise(varz,varw,s,t_p,trials)
            mass2.make_colored_noise(varz,varw,s,t_p,trials)
            mass3.make_colored_noise(varz,varw,s,t_p,trials)
            break
        
        elif experiment_number=='1B':
            # experiment 1B
            mass1 = Mass_Spring_Damper(2.5, 6, 2)
            mass2 = Mass_Spring_Damper(2.5, 6, 2)
            mass3 = Mass_Spring_Damper(2, 4, 4)
            mass1.input(2*np.cos(4*t_p)+3,'$2cos(4t_p)+3$')
            mass2.input(2*np.cos(4*t_p+3)+3,'$2cos(4t_p+3)+3$')
            mass3.input(np.sin(2*t_p),'$sin(2t_p)$')
            # set nosie
            mass1.make_colored_noise(varz,varw,s,t_p,trials)
            mass2.make_colored_noise(varz,varw,s,t_p,trials)
            mass3.make_colored_noise(varz,varw,s,t_p,trials)
            break
        
        elif experiment_number=='1C':
            # experiment 1C
            mass1 = Mass_Spring_Damper(2.5, 6, 2)
            mass2 = Mass_Spring_Damper(2.5, 6, 2)
            mass3 = Mass_Spring_Damper(2, 4, 4)
            mass1.input(2*np.cos(4*t_p)+3,'$2cos(4t_p)+3$')
            mass2.input(2*np.cos(4*t_p+3)+3,'$2cos(4t_p+3)+3$')
            mass3.input(np.sin(2*t_p),'$sin(2t_p)$')
            # set nosie
            mass1.make_colored_noise(varz,varw_noise,s,t_p,trials)
            mass2.make_colored_noise(varz,varw,s,t_p,trials)
            mass3.make_colored_noise(varz,varw,s,t_p,trials)
            break
        
        # EXPERIMENT 2: PARAMETERS FOR INCORRECT CONCLUSION
        elif experiment_number=='2A':
            # experiment 2A
            mass1 = Mass_Spring_Damper(2.5, 6, 2)
            mass2 = Mass_Spring_Damper(2.5, 6, 2)
            mass3 = Mass_Spring_Damper(2, 4, 4)
            mass1.input(2*np.cos(4*t_p)+3,'$2cos(4t_p)+3$')
            mass2.input(2*np.cos(4*t_p)+3,'$2cos(4t_p)+3$')
            mass3.input(np.sin(2*t_p),'$sin(2t_p)$')
            # set nosie
            mass1.make_colored_noise(varz,varw_noise,s,t_p,trials)
            mass2.make_colored_noise(varz,varw,s,t_p,trials)
            mass3.make_colored_noise(varz,varw,s,t_p,trials)
            break
        
        elif experiment_number=='2B':
            # experiment 2B
            mass1 = Mass_Spring_Damper(2.5, 6, 2)
            mass2 = Mass_Spring_Damper(2.25, 6, 2)
            mass3 = Mass_Spring_Damper(2, 4, 4)
            mass1.input(2*np.cos(4*t_p)+3,'$2cos(4t_p)+3$')
            mass2.input(2*np.cos(4*t_p)+3,'$2cos(4t_p)+3$')
            mass3.input(np.sin(2*t_p),'$sin(2t_p)$')
            # set nosie
            mass1.make_colored_noise(varz,varw_noise,s,t_p,trials)
            mass2.make_colored_noise(varz,varw,s,t_p,trials)
            mass3.make_colored_noise(varz,varw,s,t_p,trials)
            break
        
        elif experiment_number=='2C':
            # experiment 2C
            mass1 = Mass_Spring_Damper(2.5, 6, 2)
            mass2 = Mass_Spring_Damper(2.5, 6, 2)
            mass3 = Mass_Spring_Damper(2, 4, 4)
            mass1.input(2*np.cos(4*t_p)+3,'$2cos(4t_p)+3$')
            mass2.input(2*np.cos(4*t_p+2)+3,'$2cos(4t_p+2)+3$')
            mass3.input(np.sin(2*t_p),'$sin(2t_p)$')
            # set nosie
            mass1.make_colored_noise(varz,varw_noise,s,t_p,trials)
            mass2.make_colored_noise(varz,varw,s,t_p,trials)
            mass3.make_colored_noise(varz,varw,s,t_p,trials)
            break

    except ValueError:
        print("Error!")
'''
    w_n = np.sqrt(mass1.k/mass1.m)
    w_a = 4
    d_r = mass1.c/(2*np.sqrt(mass1.m*mass1.k))
    print('For mass m1')
    print('Natural frequency:',w_n)
    print('Input frequency:',w_a)
    print('Damping ratio:',d_r)
    '''
