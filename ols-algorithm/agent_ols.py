#!/usr/bin/env python3
'''from simulation_3-mass folder run: ./ols-algorithm/agent_ols.py
    This document finds the B matrix based on "observations" of dx and x using ordinary least squares (OLS) regression on B in a time window (delta_N). Then the script determines whether the environment is static and/or dynamic to find the agent.
    '''
import numpy as np
import matplotlib.pyplot as plt
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def polylsq_B(y, X):
    ''' To determine a from y=aX we use the matrix formula for polynomial least squares regression using a = (X^TX)^(-1) X^T y (regression in the y-direction)
        Since B is unknown and the state space equation can be rearranged as dx-Ax=Bu, polynomial OLS can be used for the state space equation if we define: y=dx-Ax, X=u, a=B
        Input:
            y   - the output of the polynomial equation
            X   - the input of the polynomial equation
        Output:
            a   - the parameters for which the OLS regression solves
        '''
    pinvX = np.linalg.pinv(X)
    a =  np.dot(pinvX, y)     # a is [b1 b2 b3 b4 b5 b6]
    return a


#--import variables from variables scripts:
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
from mass_param import n_m,experiment_number,mass1,mass2,mass3
''' Time parameters:
        trials      - number of trials of the experiment
        h           - [s] the sampling period
        T           - [s] total time
        N           - [] total number of simulation steps
        t_p         - [s] time points
        delta_N     - [] number of steps for time window
        steps       - [] range of simulation steps
'''
from time_param import trials,h,T,N,t_p,delta_N,steps


test_type = input("Enter the type of test you want to run 'full' or 'test':")
if experiment_number=='1A' or experiment_number=='1B' or experiment_number=='1C':
    if test_type== 'full':
        folder = 'correct-conclusions/exp_'
        folder_figures = 'correct-conclusions/figures_ols/'
    elif test_type== 'test':
        trials = 10
        folder = 'test_data/exp_'
        folder_figures = 'test_data/figures_ols/'
    else: print('Incorrect test type!')
elif experiment_number=='2A' or experiment_number=='2B' or experiment_number=='2C':
    if test_type== 'full':
        folder = 'incorrect-conclusions/exp_'
        folder_figures = 'incorrect-conclusions/figures_ols/'
    elif test_type== 'test':
        trials = 10
        folder = 'test_data/exp_'
        folder_figures = 'test_data/figures_ols/'
    else: print('Incorrect test type!')
folder += experiment_number+'/'
print('\nGetting data from folder:', folder)

#-- import state space and data
A = np.loadtxt(folder+'A_matrix.txt')
B_p = np.loadtxt(folder+'B_plant.txt')[...,np.newaxis]
B_a = np.loadtxt(folder+'B_agent.txt')[...,np.newaxis]
C = np.loadtxt(folder+'C_matrix.txt')
# input data
x = np.loadtxt(folder+'x.txt').reshape(N, 2*n_m, trials)
x0 = x[0,:,:]
dx = np.loadtxt(folder+'dx.txt').reshape(N, 2*n_m, trials)
y = np.loadtxt(folder+'y.txt').reshape(N, n_m, trials)
u_p = np.loadtxt(folder+'u_plant.txt')[...,np.newaxis]
# import B_i
B_i = np.loadtxt(folder+'B_i.txt').reshape(N, 2*n_m, trials)

# initialize matrices
y_hat = np.zeros(shape=y.shape)
dx_hat = np.zeros(shape=dx.shape)
dx_obshat = np.zeros(shape=dx.shape)
x_hat = np.zeros(shape=x.shape)
B_hat = np.zeros(shape=dx.shape)
B_obs = np.zeros(shape=dx.shape)
eps_dx = np.zeros(shape=dx.shape)
eps_y = np.zeros(shape=y.shape)
i_agent = np.arange(6)

#-- B at each time step using incremental polynomial OLS within a time window
# pre-calculation y of matrix formula for polynomial regression as y=dx-Ax, a=B, X=u
Y = np.zeros(shape=dx.shape)
for t in range(trials):
    for i in steps:
        Y[i,:,t] = dx[i,:,t] - A.dot(x[i,:,t])
    B_obs[0,:,t] = polylsq_B(Y[[0],:,t], mass1.u[[0]])

# pre-calculations for forward euler step
I = np.identity(A.shape[0])
Ad = I + h*A

# initial conditions
x_hat[0,:,:] = x0
dx_hat[0,:,:] = A.dot(x_hat[0,:,:]) + B_obs[0,:,:].dot(mass1.u[0,0])

# adjustable parameters
prior = False
if prior is False: save_prior = '_noprior'
else: save_prior = ''
steps_window = delta_N      # the time window of estimation of B with OLS

# collect data
t_agent = np.zeros(2*n_m)   # time of agency per state
t_env = np.zeros(2*n_m)     # time of no agency per state
cum_error = 0               # cumulative error of the agent state

print('Prior is set to:', prior)
for t in range(trials):
    #print('Trial:',t)
    t_agent_1trial = np.zeros(2*n_m)
    for i in steps:
        #print('At t=',i*h,'s')
        #--If the time window larger than the number of time steps passed the window cannot be set yet
        delta_N = steps_window     # define number of time steps over which B is determined
        if i-delta_N<0:
            delta_N = i

        #--Error, epsilon
        # calculate the average absolute error over the time window for each dx
        eps_dx[i,:,t] = (np.abs(dx_hat[i-delta_N:i+1,:,t]-dx[i-delta_N:i+1,:,t])).mean(axis=0)
        #eps_dx[i,:,t] = (np.abs(dx_hat[i-delta_N:i+1,i_agent,t]-dx[i-delta_N:i+1,:,t])).mean(axis=0)
        eps_y[i,:,t] = (np.abs(y_hat[i-delta_N:i+1,:,t]-y[i-delta_N:i+1,:,t])).mean(axis=0)
        
        #--Matrix formula for OLS
        # choose y and d within the time window
        y_d = Y[i-delta_N:i+1,:,t]
        u_d = mass1.u[i-delta_N:i+1]
        
        # update B using OLS regression over the time window
        B_obs[i,:,t] = polylsq_B(y_d, u_d)

        # set the predicted B equal to the observed B (from OLS)
        B_hat[i,:,t] = B_obs[i,:,t]
        
        #--Apply the priors if set to true
        # prep
        i_states = np.arange(0,2*n_m)               # range of all possible states
        mask = np.ones(len(i_states), dtype=bool)   # boolean mask

        if prior==True and i>0:
            #--Determine if the agent has agency or not
            # the no-agency mass is the one for which b_j is close to 0 or neg.
            i_env = np.where(B_hat[i,:,t]<0.1)[0]       # all options are not the agent
            mask[i_env] = False                         # set no agency states to false
            i_agent = i_states[mask]                    # select the state that is left
            # if there are more than one agency options for the agent
            if len(i_agent)>1:
                # then the agent is the one which is best estimated by dx
                i_agent = i_agent[np.where(eps_dx[i,i_agent,t] == eps_dx[i,i_agent,t].min())]
                # the environment is the rest
                mask = np.ones(len(i_states), dtype=bool)   # boolean mask
                mask[i_agent] = False                       # set agency states to false
                i_env = i_states[mask]                      # select the state that is left
            
            # set the no agency options to 0 (prior)
            B_hat[i,i_env,t] = 0

        else:
            i_agent = i_states
            i_env = []

        # count time for agency and no-agency and add error
        t_agent[i_agent] += h
        t_env[i_env] += h
        if len(i_agent)>0:
            cum_error += np.sum(np.abs(eps_dx[i,i_agent,t]))
        t_agent_1trial[i_agent] += h

        # prediction of next step with forward euler (except for last loop)
        if i!=(N-1):
            Bd = h*B_hat[i,:,t]
            x_hat[i+1,:,t] = Ad.dot(x[i,:,t]) + Bd*mass1.u[i,0]
            y_hat[i+1,:,t] = C.dot(x_hat[i+1,:,t])
            dx_hat[i+1,:,t] = A.dot(x_hat[i+1,:,t]) + B_hat[i,:,t].dot(mass1.u[i+1,0])

    print('Time agent after trial',t,':',t_agent_1trial)



# save data for comparison of prior and no_prior case
B_hat_mean = np.mean(B_hat,axis=(0,2))
B_hat_var = np.var(B_hat,axis=(0,2))
eps_dx_mean = np.mean(eps_dx,axis=(0,2))
eps_dx_var = np.var(eps_dx,axis=(0,2))

if prior == True:
    np.savetxt(folder+'B_hat_ols.txt',B_hat.reshape(-1,B_hat.shape[2]))
    np.savetxt(folder+'data_ols_'+experiment_number+'.txt',(B_hat_mean,eps_dx_mean,B_hat_var,eps_dx_var,t_agent))
else:
    np.savetxt(folder+'B_hat_ols_noprior.txt', B_hat.reshape(-1,B_hat.shape[2]))
    np.savetxt(folder+'data_ols_'+experiment_number+'_noprior.txt',(B_hat_mean,eps_dx_mean,B_hat_var,eps_dx_var))

print('\nThis is experiment number:',experiment_number, 'and the prior is',prior)

print('\nMEAN B MATRIX')
print('\tleast-squares:',B_hat_mean)
print('\tincremental:',np.mean(B_i,axis=(0,2)))

print('\nMEAN ERROR:')
print('\tsensory (y) prediction error:',np.mean(eps_y,axis=(0,2)))
print('\tmotion (dx) prediction error:',eps_dx_mean)

print('\nVARIANCE MATRIX')
print('\tof least-squares B_hat:',B_hat_var)
print('\tof B_i:',np.var(B_i,axis=(0,2)))
print('\tof eps_dx:',eps_dx_var)

print('\nPERCENTAGE TIME')
print('\tof agency per state:',100*(t_agent/(T*trials)),'%')
print('\tof no-agency per state:',100*(t_env/(T*trials)),'%')
print('\tcheck (agency + no-agency =',T*trials,'s):', t_agent+t_env)

print('ERROR')
print('cumulative:', cum_error)
print('average:',cum_error/(N*trials))

print('\nDATA b_2')
print('The mean of')
print('\tleast-squares b_2 w prior:',np.mean(B_hat[:,1,:]))
print('\tincremental b_2:',np.mean(B_i[:,1,:]))
print('The variance of')
print('\tleast-squares b_2 w prior:',np.var(B_hat[:,1,:]))
print('\tincremental b_2:',np.var(B_i[:,1,:]))

#--PLOT SETTINGS
u_name = [mass1.u_name,mass2.u_name,mass3.u_name]
trial_nr = 3
# font settings
SMALL_SIZE = 14
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font', size=MEDIUM_SIZE)        # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)   # fontsize of the axes title
plt.rc('xtick', labelsize=SMALL_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)   # fontsize of the tick labels
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE) # fontsize of the figure title

#--FIGURE 1: compare real solution of y calculated from OLS B
fig1 = plt.figure(1, figsize=(11, 7))
plt.subplots_adjust(hspace=0.55,top=0.9,left=0.15,right=0.85) # spacing
fig1.suptitle('Sensory observations ($\mathbf{y}$) over time for trial %s where prior is %s' %(trial_nr,prior))
for i in range(n_m):
    # mass 1: compare real solution of dx (without noise) to dx calculated from OLS B
    ax1 = plt.subplot(3,1,i+1)
    plt.title('Mass $m_%s$ with force input %s'%(i+1,u_name[i]), y=0.98)
    plt.ylabel('$y$')
    plt.xlabel('Time /s')
    plt.grid()

    # plot numerical solution for y (without noise)
    ax1.scatter(t_p, y[:,i,trial_nr],s=1,label='Observed pos. ($y_%s$)'%(2*i+1))
    # plot new y calculated from prior
    ax1.scatter(t_p, y_hat[:,i,trial_nr],s=3,label='Predicted pos. (${\haty}_%s$)'%(2*i+1))
    # plot error of sum of y per mass
    ax1.plot(t_p, eps_y[:,i,trial_nr],'--',color='k',label='$\epsilon_{{\mathbf{y}}_{m_%s}}$'%(i+1))

    # Position legend
    box = ax1.get_position()
    ax1.set_position([box.x0-box.width * 0.07, box.y0, box.width * 0.9, box.height]) # move box to left by 5% and shrink current axis by 10% (i.e. center box)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))        # put a legend to the right of the current axis
fig1.set_rasterized(True)
plt.savefig(folder_figures+'exp'+experiment_number+'_agent_y'+save_prior+'.eps', format='eps')

#--FIGURE 2: Plot the B
fig2 = plt.figure(2, figsize=(7.5, 5))
plt.subplots_adjust(hspace=1.25,top=0.85,left=0.19,right=0.90) # spacing
fig2.suptitle('$\mathbf{\hat B}$ over time for trial %s where prior is %s' %(trial_nr,prior))
for i in range(n_m):
    ax1 = plt.subplot(3,1,i+1)
    plt.title('Mass $m_%s$ with force input %s' %((i+1),u_name[i]), y=0.98)
    plt.ylabel('$b_j$')
    plt.xlabel('Time /s')
    plt.grid()
    plt.ylim(np.min((min(B_hat[:,2*i,trial_nr]),min(B_hat[:,2*i+1,trial_nr]),min(B_a[2*i]),min(B_a[2*i+1])))-0.05,np.max((max(B_hat[:,2*i,trial_nr]),max(B_hat[:,2*i+1,trial_nr]),max(B_a[2*i]),max(B_a[2*i+1])))+0.05)
    # plot numerical solution for dx (without noise)
    ax1.scatter(t_p, B_hat[:,2*i,trial_nr], label=r'$\hatb_%s$ (OLS)'%(2*i+1), color='C0', s=3)
    ax1.scatter(t_p, B_hat[:,2*i+1,trial_nr], label=r'$\hatb_%s$ (OLS)'%(2*i+2), color='C3', s=3)
    
    # plot real values of B
    plt.axhline(B_a[2*i], color='k', linestyle=':', label='$b_%s$ (true)'%(2*i+1))
    plt.axhline(B_a[2*i+1], color='k', linestyle='--', label='$b_%s$ (true)'%(2*i+2))

    # Position legend
    box = ax1.get_position()
    ax1.set_position([box.x0-box.width * 0.07, box.y0, box.width * 0.9, box.height]) # move box to left by 5% and shrink current axis by 10% (i.e. center box)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))        # put a legend to the right of the current axis
fig2.set_rasterized(True)
plt.savefig(folder_figures+'exp'+experiment_number+'_agent_b'+save_prior+'.eps', format='eps')

#--FIGURE 3: plot dx (discretized)
fig3 = plt.figure(3, figsize=(11, 7))
plt.subplots_adjust(hspace=0.60,top=0.90,left=0.15,right=0.85) # spacing
fig3.suptitle('$\mathbf{\dot{x}}$ over time for trial %s where prior is %s' %(trial_nr,prior))
for i in range(n_m):
    # mass 1: compare real solution of dx (without noise) to dx calculated from OLS B
    ax1 = plt.subplot(3,1,i+1)
    plt.title('Mass $m_%s$ with force input %s'%(i+1,u_name[i]), y=0.98)
    plt.ylabel('$\dot{x}$')
    plt.xlabel('Time /s')
    plt.grid()
    
    # plot numerical solution for dx (without noise)
    ax1.scatter(t_p, dx[:,2*i,trial_nr],s=2,label='Observed vel. ($\dot{x}_%s$)'%(2*i+1))
    ax1.scatter(t_p, dx[:,2*i+1,trial_nr],s=2,label='Observed acc. ($\dot{x}_%s$)'%(2*i+2))
    # plot new dx calculated from prior
    ax1.scatter(t_p, dx_hat[:,2*i,trial_nr],s=4,label='Predicted vel. ($\dot{\hatx}_%s$)'%(2*i+1))
    ax1.scatter(t_p, dx_hat[:,2*i+1,trial_nr],s=4,label='Predicted acc. ($\dot{\hatx}_%s$)'%(2*i+2))
    # plot error of sum of dx1 and dx2
    ax1.plot(t_p, (eps_dx[:,2*i,trial_nr]+eps_dx[:,2*i+1,trial_nr]),'--',color='k',label='$\epsilon_{\dot{x}_{%s}}+\epsilon_{\dot{x}_{%s}}$'%(2*i+1,2*i+2))
    
    # Position legend
    box = ax1.get_position()
    ax1.set_position([box.x0-box.width * 0.07, box.y0, box.width * 0.9, box.height]) # move box to left by 5% and shrink current axis by 10% (i.e. center box)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))        # put a legend to the right of the current axis
fig3.set_rasterized(True)
plt.savefig(folder_figures+'exp'+experiment_number+'_agent_dx'+save_prior+'.eps', format='eps')

'''
#--FIGURE 4: plot dx error vs lsq B
fig4 = plt.figure(4)
plt.subplots_adjust(top=0.90,right=0.95,bottom=0.12) # spacing
fig4.suptitle('OLS estimatation of $\hat{b}_{ij}$ vs. the prediction error $\epsilon_{\dot{x}_{ij}}$')
plt.grid()
for i in range(6):
    plt.ylabel('$\epsilon_{\dot{x}_{ij}}$')
    plt.xlabel('OLS $\hat{b}_{ij}$')
    plt.grid()
    
    # plot lsqB vs error in dx
    plt.scatter(B_hat[:,i,:], eps_dx[:,i], label='$j=%s$'%(i+1), s=3)
    plt.legend()
fig4.set_rasterized(True)
plt.savefig(folder_figures+'exp'+experiment_number+'_scatter_bhat_error'+save_prior+'.eps', format='eps')

#--FIGURE 5: plot dx error vs lsq B
fig5 = plt.figure(5)
plt.subplots_adjust(top=0.90,right=0.95,bottom=0.12) # spacing
fig5.suptitle('Observed ${b}_{ij}$ vs. the prediction error $\epsilon_{\dot{x}_{ij}}$')
plt.grid()
for i in range(6):
    plt.ylabel('$\epsilon_{\dot{x}_{ij}}$')
    plt.xlabel('${b}_{ij}$')
    plt.grid()
    
    # plot lsqB vs error in dx
    plt.scatter(B_i[:,i], eps_dx[:,i], label='$j=%s$'%(i+1), s=3)
    plt.legend()
fig5.set_rasterized(True)
plt.savefig(folder_figures+'exp'+experiment_number+'_scatter_b_error'+save_prior+'.eps', format='eps')
'''
#plt.show() # uncomment if you want the plot to show
