#!/usr/bin/env python3
'''./plant.py
    Using the parameters set in parameters.py, this document determines the acceleration, velocity, and displacement states for each mass. The generated data saves to the correct_conclusions or incorre_conclusions folder depending on the experiment number.
    If the input is static (i.e. u is fixed) then the numerical solution can be compared to an exact solution using odeint from the scipy.integrate package.
    '''
import numpy as np
import matplotlib.pyplot as plt

def backw_eul(A,B,C,x0,h,N_t,u):
    ''' Backward euler i.e. dx = (x_k - x_(k-1))/h
        input:
        A,B,C: continuous time system matrices
        x0: initial states of the mass nxm=1x2
        h: (time) step size [s]
        N_t: total number of time steps
        output:
        xd: vector of discritized x ([x1, x2] for N_t steps)
        yd: vector of discritized y
        '''
    I = np.identity(A.shape[0])
    Ad = np.linalg.inv(I - h*A)
    Bd = h*Ad.dot(B)
    
    xd = np.zeros(shape = (N, A.shape[0], 1))
    yd = np.zeros(shape = (N, C.shape[0], 1))
    xd[0,:,:] = x0
    
    # Step equations forward in time
    for n in range(1,N_t):
        xd[n,:,:] = Ad.dot(xd[n-1,:,:]) + Bd*(u[[n],:].T)
        yd[n,:,:] = C.dot(xd[n-1,:,:])
    
    return xd,yd

def forw_eul(A,B,C,x0,h,T,u):
    ''' Forward euler i.e. dx = (x_(k+1) - x_k)/h
        input:
        A,B,C: continuous time system matrices
        x0: initial states of the mass nxm=1x2
        h: (time) step size [s]
        N_t: total number of time steps
        output:
        xd: vector of discritized x ([x1, x2] for N_t steps)
        yd: vector of discritized y
        '''
    I = np.identity(A.shape[0])
    Ad = I + h*A
    Bd = h*B
    xd = np.zeros(shape = (N, A.shape[0], 1))
    yd = np.zeros(shape = (N, C.shape[0], 1))
    
    xd[0,:,0] = x0
    # Step equations forward in time
    for n in range(0,N_t):
        xd[n+1,:,:] = Ad.dot(xd[n,:,:]) + Bd*(u[[n],:].T)
        yd[n+1,:,:] = C.dot(xd[n,:,:])
    
    return xd,yd

def ss_msd(A, B, C, u, x, w, z):
    ''' Mass spring damper system in state space form
        dx = Ax + Bu + w
        input:
        m: mass
        k: spring constant
        c: damper constant
        F: applied force
        x: statesx1 [disp., vel.]
        output:
        dx: statesx1 [vel., accel.]'''
    x = x.reshape(A.shape[0], 1)
    u = u.reshape(A.shape[0], 1)
    dx = A.dot(x) + B*(u) + w
    y = C.dot(x) + z
    return dx,y

def mass(x,t,m,k,c,F):
    x1, x2 = x
    dxdt = [x2, -k/m*x1 - c/m*x2 + 1/m*F]
    return dxdt

def B_inc(A, u, x, dx):
    ''' Determine B according to B = (dx - A*x)/u .
        Input:
        A   - state matrix nxn
        F   - applied force
        x   - 2x1 [disp., vel.]
        dx  - 2x1 [vel., accel.]
        Output:
        B   - input matrix
        '''
    u[u==0] = 0.01  # remove 0 to avoid divide by 0
    B = (dx - A.dot(x))/u
    return B

#--import variables from variables scripts:
'''Time parameters:
    trials      - number of trials of the experiment
    h           - [s] the sampling period
    T           - [s] total time
    N           - [] total number of simulation steps
    t_p         - [s] time points
    delta_N     - [] number of steps for time window
    steps       - [] range of simulation steps
    '''
from time_param import trials,h,T,N,N_t,t_p,delta_N,steps
''' Parameters for each mass:
    x0                  - initial conditions for Euler integration of x
    n_m                 - number of mass
    experiment_number   - number of the performed experiment (1A,1B,etc.)
    state space system (i.e. dx = Ax+Bu+w, y = Cx+z)
    self.A      - state matrix
    self.B      - input matrix
    self.C      - output matrix
    self.u      - input sequence (applied force)
    self.w      - state noise
    self.z      - measurement noise
    '''
from mass_param import x0,n_m,experiment_number,mass1,mass2,mass3

#--Generate state space for the plant
A = np.zeros((n_m*mass1.A.shape[0],n_m*mass1.A.shape[0]))
A[0:2,0:2] = mass1.A
A[2:4,2:4] = mass2.A
A[4:6,4:6] = mass3.A
B_a = np.zeros((2*n_m,1))
B_a[0:2] = mass1.B
B_p = np.zeros((2*n_m,1))
B_p[0:2] = mass1.B
B_p[2:4] = mass2.B
B_p[4:6] = mass3.B
C = np.zeros((n_m*mass1.C.shape[0],n_m*mass1.C.shape[1]))
C[0,0:2] = mass1.C
C[1,2:4] = mass2.C
C[2,4:6] = mass3.C

# make one big state space matrix: input data
x0 = np.tile(x0,n_m).reshape((-1,2*n_m)).T
u_p = np.hstack((np.zeros((N,1)),mass1.u,np.zeros((N,1)),mass2.u,np.zeros((N,1)),mass3.u))
u_name = [mass1.u_name,mass2.u_name,mass3.u_name]
# noise
w = np.vstack((mass1.w,mass2.w,mass3.w))
z = np.vstack((mass1.z,mass2.z,mass3.z))

#--Find states and output of the observed plant
# forward euler to determine x and y, derivative of x, dx and output, y
x = np.zeros((N, 2*n_m, trials))
dx = np.zeros((N, 2*n_m, trials))
y = np.zeros((N, C.shape[0], trials))
for t in range(trials):
    x[:,:,[t]],y[:,:,[t]] = forw_eul(A,B_p,C,x0[:,t],h,N_t,u_p)
    dx_t = [A.dot(x[i,:,[t]].reshape(A.shape[0], 1)) + B_p*(u_p[[i],:].reshape(A.shape[0], 1)) + w[:,[i],t] for i in steps]
    dx[:,:,[t]] = np.asarray(dx_t)
    y_t = [C.dot(x[i,:,[t]].reshape(A.shape[0], 1)) + z[:,[i],t] for i in steps]
    y[:,:,[t]] = np.asarray(y_t)

#incremental B values
B_i = np.zeros(shape=dx.shape)
for t in range(trials):
    for i in steps:
        B_i[i,:,t] = B_inc(A, mass1.u[i], x[i,:,t], dx[i,:,t])

#-- SAVE DATA
if experiment_number=='1A' or experiment_number=='1B' or experiment_number=='1C':
    folder = 'correct-conclusions/exp_'
elif experiment_number=='2A' or experiment_number=='2B' or experiment_number=='2C':
    folder = 'incorrect-conclusions/exp_'

folder += experiment_number+'/'
print('Saving data in folder:', folder)
# incremental B values
np.savetxt(folder+'B_i.txt',B_i.reshape(-1,dx.shape[2]))
# true matrices
np.savetxt(folder+'A_matrix.txt',A)
np.savetxt(folder+'B_plant.txt',B_p)
np.savetxt(folder+'B_agent.txt',B_a)
np.savetxt(folder+'C_matrix.txt',C)
# input data
np.savetxt(folder+'dx.txt',dx.reshape(-1,dx.shape[2]))
np.savetxt(folder+'x.txt',x.reshape(-1,dx.shape[2]))
np.savetxt(folder+'y.txt',y.reshape(-1,dx.shape[2]))
np.savetxt(folder+'u_plant.txt',u_p)

#--PLOT
# dx and x
fig1 = plt.figure(1)
fig1.suptitle('Plant $\mathbf{x}$ and $\mathbf{\dot{x}}$'f'\n with h={h} s, T={T} s')
for i in range(n_m):
    plt.subplot(3,1,i+1)
    plt.xlabel('Time (in s)')
    plt.ylabel('$\mathbf{\dot{x}}_{m_%s}$ and $\mathbf{x}_{m_%s}$'%((i+1),(i+1)))
    plt.grid()
    # plot numerical solution
    plt.plot(t_p, x[:,2*i,0],'g',label='$x_%s$'%(2*i+1))
    plt.plot(t_p, x[:,2*i+1,0],'b',label='$x_%s$'%(2*i+2))
    plt.plot(t_p, dx[:,2*i,0],label='$\dot{x}_%s$'%(2*i+1))
    plt.plot(t_p, dx[:,2*i+1,0],label='$\dot{x}_%s$'%(2*i+2))
    # input
    plt.plot(t_p, u_p[:,2*i+1],'y:',label='$u$=%s'%u_name[i])
    plt.legend(loc='right')

# y
fig2 = plt.figure(2)
fig2.suptitle('Plant output, $\mathbf{y}$,'f'with h={h} s, T={T} s')
plt.xlabel('Time (in s)')
plt.ylabel('$\mathbf{y}$')
plt.grid()
for i in range(n_m):
    # plot numerical solution
    plt.plot(t_p, y[:,i,0],label='$y_%s$'%(i+1))
    # input
    plt.legend(loc='right')

# compare dx
fig3 = plt.figure(3)
fig3.suptitle('Plant $\mathbf{\dot{x}}$,'f'with h={h} s, T={T} s')
plt.xlabel('Time (in s)')
plt.ylabel('$\mathbf{\dot{x}}$')
plt.grid()
for i in range(2*n_m):
    # plot numerical solution
    plt.plot(t_p, dx[:,i,0],label='$\dot{x}_%s$'%(i+1))
    # input
    plt.legend(loc='right')

# B_i
fig4 = plt.figure(4)
fig4.suptitle('$\mathbf{B}$,'f'with h={h} s, T={T} s')
plt.xlabel('Time (in s)')
plt.ylabel('$\mathbf{\dot{x}}$')
plt.plot(t_p,mass1.u,':k',alpha=0.5,label='$u$')
plt.grid()
for i in range(2*n_m):
    # plot numerical solution
    plt.scatter(t_p, B_i[:,i,0],label='$b_%s$'%(i+1),s=2)
    # input
    plt.legend(loc='right')


#plt.show() # uncomment if you want the plot to show

