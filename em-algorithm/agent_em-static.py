#!/usr/bin/env python3
'''from simulation_3-mass folder run: ./em-algorithm/agent_em-static.py'''
from scipy import stats
#from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def e_step(pi_k, mu_k, sigma_k, x_i):
    """The expectation step: compute the probability each data point is a result of cluster k, P(k|xi)"""
    # find the likelihood P(x_i|k), dim: Kxn
    N_k = stats.norm(loc=mu_k,scale=sigma_k).pdf(x_i)
    #N_k[:,np.where(np.sum(N_k,axis=0)==0)]=0.001    # if 0 then set to 0.001 to avoid divide by zero
    
    # find the marginal likelihood P(x_i), dim: nx
    r_evidence = (np.sum(pi_k * N_k,axis=0))

    # find the posterior P(k|xi) or responsibility
    r_ik = [pi_k[k,:] * N_k[k,:] / r_evidence for k in range(len(mu_k))]
    r_ik = np.array(r_ik).T
    '''
    N_k = np.zeros((x.shape[1],pi_k.shape[0],x.shape[0]))   # dim: n x K x N
    L_k = np.zeros((x.shape[1],x.shape[0]))                 # dim: n x N
    r_k = np.zeros((x.shape[0],x.shape[1],pi_k.shape[0]))   # dim: N x n x K
    for j in range(x.shape[1]):
        N_k[j,:,:] = stats.norm(loc=mu_k,scale=sigma_k).pdf(x[:,j])
        L_k[j,:] = np.dot(pi_k[:,[j]].T,N_k[j,:,:])
        r_k[:,j,:] = np.array([pi_k[k,j] * N_k[j,k,:] / L_k[j,:] for k in range(len(mu_k))]).T
    '''
    return r_ik

def m_step(r_ik, x):
    '''The maximization step: update parameters values'''
    # sum of r_ik (posterior) over i shape Nk [Kx1] for denominator of mean and std
    Nk = np.sum(np.sum(r_ik,axis=1),axis=0)
    # update mean (general EM)
    mu_k = [np.sum(r_ik[:,:,k]*x)/Nk[k] for k in range(r_ik.shape[2])]
    mu_k = np.array([mu_k]).T
    # update std
    var_k = [np.sum(r_ik[:,:,k]*(x-mu_k[k,:])**2)/Nk[k] for k in range(r_ik.shape[2])]
    sigma_k = np.array([np.sqrt(var_k)]).T
    # update mixture proportions of the clusters (should actually be mean but this allows us to enforce prior)
    pi_k = np.sum(r_ik,axis=0).T
    # enforce prior on agent
    pi_k[0,:] = pi_k[0,:]/Nk[0]
    pi_k[1,:] = 1-pi_k[0,:]

    return mu_k, sigma_k, pi_k

def log_likelihood(pi_k, mu_k, sigma_k, x):
    '''The log-likelihood'''
    # calculate the likelihood
    N_k = np.zeros((x.shape[1],pi_k.shape[0],x.shape[0]))   # dim: n x K x N
    L_k = np.zeros((x.shape[1],x.shape[0]))                 # dim: n x N
    for j in range(x.shape[1]):
        N_k[j,:,:] = stats.norm(loc=mu_k,scale=sigma_k).pdf(x[:,j])
        L_k[j,:] = np.dot(pi_k[:,[j]].T,N_k[j,:,:])
    # calculate the log-likelihood
    L_tot = np.sum(np.log(L_k))
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
from time_param import trials,h,T,N,t_p,delta_N,steps


test_type = input("Enter the type of test you want to run 'full' or 'test':")
if experiment_number=='1A' or experiment_number=='1B' or experiment_number=='1C':
    if test_type== 'full': folder = 'correct-conclusions/exp_'
    elif test_type== 'test':
        trials = 10
        folder = 'test_data/exp_'
    else: print('Incorrect test type!')
    folder_figures = 'correct-conclusions/figures_em/'
elif experiment_number=='2A' or experiment_number=='2B' or experiment_number=='2C':
    if test_type== 'full': folder = 'incorrect-conclusions/exp_'
    elif test_type== 'test':
        trials = 10
        folder = 'test_data/exp_'
    else: print('Incorrect test type!')
    folder_figures = 'incorrect-conclusions/figures_em/'
folder += experiment_number+'/'
print('\nGetting data from folder:', folder)

#-- import state space and data
A = np.loadtxt(folder+'A_matrix.txt')
B_p = np.loadtxt(folder+'B_plant.txt')[...,np.newaxis]
B_a = np.loadtxt(folder+'B_agent.txt')[...,np.newaxis]
C = np.loadtxt(folder+'C_matrix.txt')
# import B_i
B_i = np.loadtxt(folder+'B_i.txt').reshape(N, 2*n_m, trials)
# import B_hat from OLS
#ols_B_hat = np.loadtxt(folder+'B_hat_ols.txt').reshape(N, 2*n_m, trials)

#--Set initial conditions
n_n = n_m*2                     # number of states
n_k = 2                         # number of clusters
mu = np.zeros((n_k,1))
mu[0] = 0.1                     # mean of agency
mu[1] = 0                       # mean of no-agency
mu_init = mu
sigma = np.zeros((n_k,1))
sigma[0] = np.sqrt(10**(-3))      # standard deviation of agency
sigma[1] = np.sqrt(10**(-3))      # standard deviation of no-agency
sigma_init = sigma
pi = np.zeros((n_k,n_n))
pi[0,:] = 1/n_n                 # prior of agency
pi[1,:] = (n_n-1)/n_n           # prior of no-agency
pi_init = pi

# initialize matrices for storing data
r_i = np.zeros((N,n_n,n_k,trials))      # responsibility
B_hat = np.zeros(shape=B_i.shape)       # estimation of B matrix
agency = np.zeros((N,trials))           # N means of agency cluster
env = np.zeros((N,trials))              # N means of no-agency cluster

# convergence criteria
max_iter = 5    # maximum number of iteration steps allowed
tol = 0.001     # difference in the log-likelihood
steps_window = delta_N

# perform EM for each trial step
for t in range(3):
    #print('Trial:',t)
    # reset initial conditions for each trial
    mu = mu_init
    sigma = sigma_init
    pi = pi_init
    t_agency_1trial = np.zeros(2*n_m)
    # per trial run all time steps
    for i in steps:
        #print('Time step:',i*h,'s')
        # set window boundary
        delta_N = steps_window
        if i-delta_N<0:
            delta_N = i
        # find the log-likelihood
        ll_new = log_likelihood(pi, mu, sigma, B_i[i-delta_N:i+1,:,t])

        # run EM until convergence
        for it in range(max_iter):
            ll_old = ll_new

            # E: expectation step
            r_i[i,:,:,t] = e_step(pi, mu, sigma, B_i[i,:,t])

            # M: maximization step
            mu, sigma_temp, pi = m_step(r_i[i-delta_N:i+1,:,:,t], B_i[i-delta_N:i+1,:,t])

            # only calculate the standard dev. when enough time steps passed
            if i>10:
                sigma = sigma_temp

            # check exit condition
            ll_new = log_likelihood(pi, mu, sigma, B_i[i-delta_N:i+1,:,t])
            diff_ll = abs(ll_old-ll_new)
            #print('\titeration:',it,'w log-likelihood difference:',abs(ll_old-ll_new))
            if (diff_ll<tol) or np.isnan(diff_ll)==True:
                break

        # estimate B_hat using agency mu and pi
        B_hat[i,:,t] = r_i[i,:,0,t]*mu[0]

        # save mean at each time step as b value of agency and no-agency (env)
        agency[i,t] = mu[0]
        env[i,t] = mu[1]
        t_agency_1trial += h*pi[0,:]
        
    print('Time agency after trial',t,':',t_agency_1trial)

# save data for comparison of prior and no_prior case
B_hat_mean = np.mean(B_hat,axis=(0,2))
B_hat_var = np.var(B_hat,axis=(0,2))
np.savetxt(folder+'em_B_hat_static.txt',B_hat.reshape(-1,B_hat.shape[2]))
np.savetxt(folder+'data_em_static.txt',(B_hat_mean,B_hat_var))

#--PRINT: some values for comparison
print('\nThe responsibility, r_i:')
print(r_i[i,:,0,t])

print('\nTHE MU OF:')
print('\tthe agency cluster:', np.mean(agency))
print('\tthe no-agency cluster:', np.mean(env))

print('\nTHE MEAN OF:')
print('\tEM B_hat:', np.mean(B_hat,axis=(0,2)))
#print('\tOLS B_hat:', np.mean(ols_B_hat,axis=(0,2)))
print('\tincremental B is:', np.mean(B_i,axis=(0,2)))

print('\nTHE VARIANCE OF:')
print('\tthe mu of the agency cluster:', np.var(agency))
print('\tthe mu of the no-agency cluster:', np.var(env))
print('\tEM B_hat:', np.var(B_hat,axis=(0,2)))
#print('\tOLS B_hat:', np.var(ols_B_hat,axis=(0,2)))
print('\tincremental B is:', np.var(B_i,axis=(0,2)))

print('\nFinal mean of the agency is:', mu[0], 'and of the no-agency is', mu[1])
print('Final std of the agency is:', sigma[0], 'and of the no-agency is', sigma[1])
print('Final mixture (pi) of the agency is:', pi[0], 'and of the no-agency is', pi[1])
#aprint('Final mean of OLS B_hat mass m_1 is:', ols_B_hat[-1,1,t])

#--FIGURE SETTINGS
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 16
plt.rc('font', size=MEDIUM_SIZE)        # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)   # fontsize of the axes title
plt.rc('xtick', labelsize=SMALL_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)   # fontsize of the tick labels
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE) # fontsize of the figure title
trial = t

#--FIGURE 1: histogram of B values
data_range = np.arange(-1.0,1.0,0.0001)
N1_init = (np.sum(pi_init[0,:])/6) * stats.norm(loc=mu_init[0],scale=sigma_init[0]).pdf(data_range)
N2_init = (np.sum(pi_init[1,:])/6) * stats.norm(loc=mu_init[1],scale=sigma_init[1]).pdf(data_range)
N1_fin = (np.sum(pi[0,:])/6) * stats.norm(loc=mu[0],scale=sigma[0]).pdf(data_range)
N2_fin = (np.sum(pi[1,:])/6) * stats.norm(loc=mu[1],scale=sigma[1]).pdf(data_range)

fig1 = plt.figure(1)
fig1.suptitle('Histogram of $\mathbf{X}_B$ for trial %s where \n $F_{int}=$%s, $F_{ext,1}=$%s, and $F_{ext,2}=$%s'%(trial, mass1.u_name,mass2.u_name,mass3.u_name))
plt.hist(B_i[-delta_N:,:,trial].flatten(),density=True,bins=100,label=['$b_{ij}$'])#,'$b_{i2}$','$b_{i3}$','$b_{i4}$','$b_{i5}$','$b_{i6}$'])
#plt.plot(data_range,N1_init,color='C1',label='$f(b_{ij} \mid \mu_{0}, \sigma_{0}^{2})$ at $t=0$ s',linewidth=3,ls='--')
#plt.plot(data_range,N2_init,color='C3',label='$f(b_{ij} \mid \mu_{1}, \sigma_{1}^{2})$ at $t=0$ s',linewidth=3,ls='--')
#plt.plot(data_range,N1_fin,color='C1',label='$f(b_{ij} \mid \mu_{0}, \sigma_{0}^{2})$ at $t=20$ s',linewidth=3)
#plt.plot(data_range,N2_fin,color='C3',label='$f(b_{ij} \mid \mu_{1}, \sigma_{1}^{2})$ at $t=20$ s',linewidth=3)
plt.ylabel('Frequency of $b_{ij}$')
plt.xlabel('$b_{ij}$')
plt.legend(loc='upper center')
plt.xlim(np.min(B_i[:,:,trial].flatten())-0.01,np.max(B_i[:,:,trial].flatten())+0.01)
plt.grid()

#--FIGURE 2: plot of the probability that the b_k values are the agency
u_list = [mass1.u_name, mass2.u_name, mass3.u_name]

fig2 = plt.figure(2)
plt.subplots_adjust(hspace=1,left=0.12,right=0.99) # spacing
fig2.suptitle('Probability that $b_{ij}$ is a result of agency ($Z_i=0$) for trial %s'%trial)
for i in range(0,n_m):
    # Mass w/o prior
    ax1 = plt.subplot(3,1,i+1)
    plt.title('Mass $m_%s$ with input $u=$%s' %((i+1),u_list[i]))
    plt.ylabel('$r_i(Z_i=0)$')
    plt.xlabel('Time /s')
    plt.scatter(t_p,r_i[:,2*i,0,trial],s=3,label='$j=%s$'%(2*i+1))
    plt.scatter(t_p,r_i[:,2*i+1,0,trial],s=3,label='$j=%s$'%(2*i+2))
    plt.legend(loc='upper right')


#--FIGURE 3: the b values with the prior applied
fig3 = plt.figure(3)
plt.subplots_adjust(hspace=0.49,top=0.9,left=0.08,right=0.85) # spacing
fig3.suptitle('$\mathbf{\hat B}$ over time for trial %s' %(trial))
for i in range(0,n_m):
    ax1 = plt.subplot(3,1,i+1)
    plt.title('Mass $m_%s$ with input $u=$%s' %((i+1),u_list[i]))
    plt.ylabel('$\mathbf{B}$')
    plt.xlabel('Time /s')
    # plot real B values
    plt.ylim(B_a[2*i]-0.05,B_a[2*i+1]+0.05)
    plt.axhline(B_a[2*i], color='k', linestyle=':', label='True $b_%s$'%(2*i+1))
    plt.axhline(B_a[2*i+1], color='k', linestyle=':', label='True $b_%s$'%(2*i+2))
    # plot ols_B_hat and ols_B_hat within confidence interval
    plt.scatter(t_p, B_i[:,2*i,trial], color='C0',s=4, label='$b_{i%s}$'%(2*i+2))
    plt.scatter(t_p, B_i[:,2*i+1,trial], color='C1',s=4, label='$b_{i%s}$'%(2*i+2))
    #plt.scatter(t_p, ols_B_hat[:,2*i,trial], color='C0',s=5,label='LSQ $\hatb_{%s}[i]$'%(2*i+1))
    #plt.scatter(t_p, ols_B_hat[:,2*i+1,trial], color='C1',s=5,label='LSQ $\hatb_{%s}[i]$'%(2*i+2))
    plt.scatter(t_p, B_hat[:,2*i,trial], color='C3',s=10, label='EM $\hatb_{i%s}$'%(2*i+1))
    plt.scatter(t_p, B_hat[:,2*i+1,trial], color='C2',s=10, label='EM $\hatb_{i%s}$'%(2*i+2))

    # Position legend
    box = ax1.get_position()
    ax1.set_position([box.x0-box.width * 0.07, box.y0, box.width * 0.9, box.height]) # move box to left by 5% and shrink current axis by 10% (i.e. center box)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))        # put a legend to the right of the current axis
plt.show()
