#!/usr/bin/env python3
'''from simulation_3-mass folder run: ./em-algorithm/agent_em-dyn.py
    Make sure to run ./plant.py and ./ols-algorithm/agent_ols.py before to ensure that the correct data is being used and plotted.
    '''
from scipy import stats
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def e_step(pi_k, mu_k, cov_k, x_i):
    """The expectation step: compute the probability each data point is a result of cluster k, P(k|xi)"""
    
    # find the likelihood P(x_i|k), dim: Kxn
    N_k = np.zeros(shape=(mu_k.shape[0],x_i.shape[0]))
    for k in range(mu_k.shape[0]):
        N_k[k,:] = stats.multivariate_normal.pdf(x_i,mean=mu_k[k,:],cov=cov_k[k,:])
    if np.any(N_k==0): N_k[np.where(N_k==0)]=1e-6
    
    # find the marginal likelihood P(x_i), dim: n
    r_evidence = (np.sum(pi_k * N_k,axis=0))

    # find the posterior P(k|xi) or responsibility, dim:Kxn
    r_ik = [(pi_k[k,:] * N_k[k,:]) / r_evidence for k in range(mu_k.shape[0])]
    r_ik = np.array(r_ik)
    
    return r_ik

def m_step(r_ik, x, i_agent, i_env):
    '''The maximization step'''
    
    # sum of r_ik (posterior) over i shape Nk [Kx1] for denominator of mean and std
    r_ik = np.nan_to_num(r_ik)
    Nk = np.sum(np.sum(r_ik,axis=1),axis=1)
    Nk[np.where(Nk==0)] = 1e-5
    
    # mixture proportions without prior
    #pi_k = np.mean(r_ik,axis=1)
    # mixture proportions with prior
    pi_k = np.sum(r_ik,axis=1) / Nk[i_agent]
    remainder = 1 - pi_k[i_agent,:]
    sumN_rik = np.sum(r_ik,axis=1)
    scale = remainder / (np.sum(sumN_rik[i_env],axis=0))
    scale = np.nan_to_num(scale)
    pi_k[i_env,:] = sumN_rik[i_env]*scale
    #print('The agent control 1 mass:',np.sum(pi_k,axis=1)[0]==1,np.sum(pi_k,axis=1)[0])
    
    # update mean
    r_ik = r_ik.reshape(r_ik.shape[0], r_ik.shape[1]*r_ik.shape[2])
    x = x.reshape(x.shape[0]*x.shape[1], x.shape[2])
    mu_k = [np.dot(r_ik[k,:], x) / Nk[k] for k in range(r_ik.shape[0])]
    mu_k = np.vstack(mu_k)
    
    # update covariance matrix
    cov_k = [np.dot((r_ik[[k],:].T*(x - mu_k[k,:])).T, (x - mu_k[k,:])) / Nk[k] for k in range(r_ik.shape[0])]
    cov_k = np.array(cov_k)
    cov_k = cov_k + 1e-5*np.identity(len(x[0]))

    return mu_k, cov_k, pi_k

def log_likelihood(pi_k, mu_k, cov_k, x):
    '''The log-likelihood'''
    '''
    # calculate the likelihood for entire time window
    N_k = np.zeros((x.shape[1],mu_k.shape[0],x.shape[0]))   # dim: n x K x N
    L_k = np.zeros((x.shape[1],x.shape[0]))                 # dim: n x N
    for j in range(x.shape[1]):
        for k in range(mu_k.shape[0]):
            N_k[j,k,:] = stats.multivariate_normal.pdf(x[:,j],mean=mu_k[k,:],cov=cov_k[k,:])
        L_k[j,:] = np.dot(pi_k[:,[j]].T,N_k[j,:,:])
    if np.any(N_k==0): N_k[np.where(N_k==0)]=1e-6
    
    # calculate the log-likelihood
    L_tot = np.sum(np.log(L_k))
    '''
    # calculate the likelihood for one data point
    N_k = np.zeros(shape=(mu_k.shape[0],x.shape[0]))
    for k in range(mu_k.shape[0]):
        N_k[k,:] = stats.multivariate_normal.pdf(x,mean=mu_k[k,:],cov=cov_k[k,:])
    
    # calculate the log-likelihood
    L_tot = np.sum(np.log(np.sum(pi_k * N_k,axis=0)))

    return L_tot

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
from time_param import trials,h,T,N,t_p,delta_N,steps,N_t

test_type = input("Enter the type of test you want to run 'full' or 'test':")
if experiment_number=='1A' or experiment_number=='1B' or experiment_number=='1C':
    if test_type== 'full': folder = 'correct-conclusions/exp_'
    elif test_type== 'test':
        trials = 10
        folder = 'test_data/exp_'
    else: print('Incorrect test type!')
    folder_figures = 'correct-conclusions/figures_em_cov/'
elif experiment_number=='2A' or experiment_number=='2B' or experiment_number=='2C':
    if test_type== 'full': folder = 'incorrect-conclusions/exp_'
    elif test_type== 'test':
        trials = 10
        folder = 'test_data/exp_'
    else: print('Incorrect test type!')
    folder_figures = 'incorrect-conclusions/figures_em_cov/'
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
# import B_hat from OLS
ols_B_hat = np.loadtxt(folder+'B_hat_ols_noprior.txt').reshape(N, 2*n_m, trials)

#--Set initial conditions
n_n = n_m*2             # number of states
n_k = int((n_n/2)+2)    # number of clusters
n_x = 2                 # number of dimensions
# mean, mu
mu = np.zeros((n_k,n_x))
mu[:,0] = [0.5,0.0,0.5,0.5,0.5]         # k means for b_{ij}
mu[:,1] = [0.1,0.0,1.0,1.5,0.5]         # k means for error in dx
mu_init = mu
# covariance matrix (C_{i,j} = sigma(x_i,x_j))
cov = np.zeros((n_k,n_x,n_x))                       # cov. matrices for each cluster
cov[0,:,:] = [[10**(-3), 0], [0, 10**(-3)]]         # cov. matrix agency
cov[1,:,:] = [[10**(-3), 0], [0, 10**(-3)]]         # cov. matrix no-agency (static)
cov[2,:,:] = [[10**(-1), 0], [0, 10**(-2)]]         # cov. matrix no-agency (dynamic)
cov[3,:,:] = [[10**(-1), 0], [0, 10**(-2)]]         # cov. matrix no-agency (dynamic)
cov[4,:,:] = [[10**(-1), 0], [0, 10**(-2)]]         # cov. matrix no-agency (dynamic)
cov_init = cov
# mixture proportions of the clusters (i.e. the prior), pi
pi = np.zeros((n_k,n_n))
pi[0,:] = 1/6         # prior of agent
pi[1,:] = 2/6         # prior of static env.
pi[2,:] = 1/6         # prior of dynamic env.
pi[3,:] = 1/6         # prior of dynamic env.
pi[4,:] = 1/6         # prior of dynamic env.
pi_init = pi
# test different IC
cov_test = np.logspace(-6,-1,15)
#pi_test = np.linspace(0,1,7)
n_c = len(cov_test)

#--Initialize matrices to store data for all time steps
data = np.zeros((N,n_n,n_x))            # data, X
r_i = np.zeros((n_k,N,n_n,trials))      # responsibility
x_hat = np.zeros(shape=x.shape)         # prediction of states
x_hat[0,:,:] = x0                       # initial condition x
y_hat = np.zeros(shape=y.shape)         # prediction of observations
dx_hat = np.zeros(shape=dx.shape)       # prediction of motion states
dx_hat[0,:,:] = dx[0,:,:]               # initial condition dx
eps_x = np.zeros(shape=x.shape)         # prediction error in states
eps_y = np.zeros(shape=y.shape)         # prediction error in observations
eps_dx = np.zeros((N, 2*n_m,trials,n_c))       # prediction error in motion states
B_hat =  np.zeros((N, 2*n_m,trials,n_c))       # agency estimation (EM and prior adjusted)

# stores data of agency clusters
mu_agency = np.zeros((N,n_x,trials,n_c))
pi_agency = np.zeros((N,n_n,trials,n_c))
cov_agency = np.zeros((N,trials,n_c))
t_agency = np.zeros(2*n_m)          # time of agency per state
i_agency = np.zeros((N,n_n,trials))

# pre-calculations for forward euler step
I = np.identity(A.shape[0])
Ad = I + h*A

# iteration specifications of EM
max_iter = 5
tol = 0.001
steps_window = delta_N

#--Perform EM for each trial
for c in range(n_c):
    cov_init[0,0,0] = cov_test[c]
    cov_init[0,1,1] = cov_test[c]
    print('\nCovariance', c, ':',cov_init[0,0,0])
    #pi_init[0,:] = pi_test[c]
    #pi_init[1:5,:] = (1-pi_test[c])/(n_k-1)
    #print('\nPi', c, ':',pi_init[0,:])
    for t in range(trials):
        # reset initial conditions for each trial
        mu = mu_init
        cov = cov_init
        pi = pi_init
        i_agent = 0
        i_env = np.arange(1,n_k)
        t_agent_1trial = np.zeros(2*n_m)
        # per trial, run all time steps
        for i in steps:
            #print('\nTime step:',i*h,'s')
            # set steps window
            delta_N = steps_window
            if i-delta_N<0:
                delta_N = i

            # determine prediction error for states and observations
            eps_dx[i,:,t,c] = (np.abs(dx_hat[i-delta_N:i+1,:,t]-dx[i-delta_N:i+1,:,t])).mean(axis=0)
            #eps_y[i,:,t] = (np.abs(y_hat[i-delta_N:i+1,:,t]-y[i-delta_N:i+1,:,t])).mean(axis=0)
            
            # save the data at this time step in data (i.e. X)
            data[i,:,0] = B_i[i,:,t]
            data[i,:,1] = eps_dx[i,:,t,c]

            # calculate the log-likelihood
            ll_new = log_likelihood(pi, mu, cov, data[i,:,:])

            # run EM until convergence or max number of iterations
            for j in range(max_iter):
                ll_old = ll_new
                
                # E: expectation step
                r_i[:,i,:,t] = e_step(pi, mu, cov, data[i,:,:])
                # M: maximization step
                mu, cov_temp, pi = m_step(r_i[:,i-delta_N:i+1,:,t], data[i-delta_N:i+1,:,:],i_agent,i_env)

                # calculate the covariance and mixtures when there are enough data points (enforce initial conditions for some time steps)
                if i>10: cov = cov_temp
                
                # check exit condition
                ll_new = log_likelihood(pi, mu, cov, data[i,:,:])
                diff_ll = abs(ll_old-ll_new)
                #print('\titeration:',j,'w log-likelihood difference:',abs(ll_old-ll_new))
                if (diff_ll<tol) or np.isnan(diff_ll)==True:
                    break
            
            # if enough data is collected, find the agency cluster
            if i>10:
                i_k = np.arange(0,n_k)                      # range of all possible clusters
                mask = np.ones(len(i_k), dtype=bool)        # boolean mask
                static = np.where(mu[:,0]<0.1)[0]           # static cluster
                mask[static] = False                        # no agency over static cluster
                i_agent = i_k[mask]                         # remove static cluster from agency options
                # while more than one possible agency cluster
                while len(i_agent)>1:
                    max_cov = i_agent[np.argmax(cov[i_agent,0,0])]  # find maximum covariance
                    mask[max_cov] = False   # no agency for large covariance
                    i_agent = i_k[mask]     # remove no agency cluster from agency options
                # if no agency options are left
                if len(i_agent)==0:
                    i_agent = [np.argmax(mu[:,0])]          # find the maximum mean for bij
                    mask[i_agent] = True                    # this is the agency cluster
                i_agent = i_agent[0]
                i_env = i_k[~mask]        # no-agency clusters are opposite of agency cluster

            # agency parameter, B_hat, with mu and r_i for x_hat and dx_hat
            B_hat[i,:,t,c] = np.dot(pi[[i_agent],:].T,mu[i_agent,[0]])

            # prediction of next step with forward euler (except for last loop)
            if i!=(N-1):
                Bd = h*B_hat[i,:,t,c]
                x_hat[i+1,:,t] = Ad.dot(x[i,:,t]) + Bd*mass1.u[i,0]
                dx_hat[i+1,:,t] = A.dot(x_hat[i+1,:,t]) + B_hat[i,:,t,c].dot(mass1.u[i+1,0])

            # store the agency data for each time step
            t_agency += h*pi[i_agent,:]
            t_agent_1trial += h*pi[i_agent,:]
            mu_agency[i,:,t,c] = mu[i_agent,:]
            cov_agency[i,t,c] = cov[i_agent,0,0]
            pi_agency[i,:,t,c] = pi[i_agent,:]
            i_agency[i,:,t] = i_agent
        print('Time agent after trial',t,':',t_agent_1trial)

#--PLOT
trial = 3
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

# B_hat (N, 2*n_m, trials, n_c)
B_hat_mean_t = np.mean(B_hat,axis=(2))                      # N, 2*n_m, n_c
B_hat_meanN = np.mean(B_hat,axis=(0))                       # 2*n_m, trials, n_c
b2_meanN_mean = np.mean(B_hat_meanN[1,:,:],axis=0)          # n_c
B_hat_varN = np.var(B_hat,axis=(0))                         # 2*n_m, trials, n_c
b2_varN_mean = np.mean(B_hat_varN[1,:,:],axis=0)            # n_c
error_b2 = np.abs(B_hat_mean_t[:,1,:]-0.4)                  # N, n_c
mu_mean_t = np.abs(np.mean(mu_agency[:,0,:,:],axis=1)-0.4)  # N, n_c
cov_a_nc = np.mean(cov_agency,axis=1)                       # N, n_c
error_b2_ols = np.abs(np.mean(ols_B_hat[:,1,:])-0.4)
var_b2_ols = np.var(ols_B_hat[:,1,:])

cov_power = np.linspace(-6,-1,n_c)
#cov_power = cov_test
colors_trial = plt.cm.tab10(np.linspace(0,1,trials))
colors_cov = plt.cm.coolwarm(np.linspace(0,1,n_c))

fig1 = plt.figure(1, figsize=(11, 5))
plt.subplots_adjust(left=0.1,bottom=0.12,right=0.97,top=0.92)
plt.title('Error in $\hat{b}_{i2}$ for each initial covariance matrix per trial')
plt.xlabel('Initial covariance')
plt.ylabel('Mean error in $\hat{b}_{i2}$')
plt.xscale('log')
for t in range(trials):
    plt.plot(cov_test, np.abs(B_hat_meanN[1,t,:]-0.4), c=colors_trial[t], label='Trial %s'%(t+1))
plt.plot(cov_test, np.abs(b2_meanN_mean-0.4), ls='--',c='k', label='Mean (EM)')
plt.axhline(error_b2_ols, ls=':', c='k',label='Mean (OLS)')
plt.legend(loc='upper left',ncol=2)
plt.grid()
fig1.set_rasterized(True)
plt.savefig(folder_figures+'exp'+experiment_number+'_errorb2-vs-cov.eps', format='eps')

fig2 = plt.figure(2, figsize=(11, 5))
plt.subplots_adjust(left=0.1,bottom=0.12,right=0.97,top=0.92)
plt.title('Variance in $\hat{b}_{i2}$ for each initial covariance matrix per trial')
plt.xlabel('Initial covariance')
plt.ylabel('Mean variance in $\hat{b}_{i2}$')
plt.xscale('log')
for t in range(trials):
    plt.plot(cov_test, B_hat_varN[1,t,:], c=colors_trial[t], label='Trial %s'%(t+1))
plt.plot(cov_test, b2_varN_mean, ls='--',c='k', label='Mean (EM)')
plt.axhline(var_b2_ols, ls=':', c='k',label='Mean (OLS)')
plt.legend(loc='upper left',ncol=2)
plt.grid()
fig1.set_rasterized(True)
plt.savefig(folder_figures+'exp'+experiment_number+'_varb2-vs-cov.eps', format='eps')

fig3 = plt.figure(3)
plt.subplots_adjust(right=0.97,top=0.94, left=0.15)
plt.title('Variance of agency cluster per data point')
plt.xlabel('Time \s')
plt.ylabel('Variance')
for c in range(n_c):
    plt.scatter(t_p, cov_a_nc[:,c], c=colors_cov[c], s=5, label='$10^{%0.1f}$'%(cov_power[c]))
plt.legend(ncol=2)
plt.grid()

fig4 = plt.figure(4)
plt.subplots_adjust(right=0.97,top=0.94)
plt.title('Error in $b_2$ over time')
plt.ylabel('Error in $b_2$')
plt.xlabel('Time \s')
for c in range(n_c):
    plt.scatter(t_p, error_b2[:,c], c=colors_cov[c], s=5, label='$10^{%0.1f}$'%(cov_power[c]))
plt.legend(ncol=2)
plt.grid()

fig5 = plt.figure(5)
plt.subplots_adjust(right=0.97,top=0.94)
plt.title('Error in $\mu$ of agency cluster, $Z_i=%s$'%(i_agent))
plt.xlabel('Time \s')
plt.ylabel('Error in $\mu$')
for c in range(n_c):
    plt.scatter(t_p, mu_mean_t[:,c], c=colors_cov[c], s=5, label='$10^{%0.1f}$'%(cov_power[c]))
plt.legend(ncol=2)
plt.grid()

plt.show() # uncomment if you want to plot

#--SAVE DATA FOR COMPARISON
print('Saving data...')
B_hat_mean = np.mean(B_hat,axis=(0,2))      #2*n_m, n_c
B_hat_var = np.var(B_hat,axis=(0,2))        #2*n_m, n_c
eps_dx_mean = np.mean(eps_dx,axis=(0,2))    #2*n_m, n_c
eps_dx_var = np.var(eps_dx,axis=(0,2))      #2*n_m, n_c
data = np.vstack((B_hat_mean, eps_dx_mean, B_hat_var, eps_dx_var))
np.savetxt(folder+'data_em_cov'+experiment_number+'.txt', data, delimiter = ',')

