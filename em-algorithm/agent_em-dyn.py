#!/usr/bin/env python3
'''from simulation_3-mass folder run: ./em-algorithm/agent_em-dyn.py
    Make sure to run ./plant.py and ./ols-algorithm/agent_ols.py before to ensure that the correct data is being used and plotted.
    '''
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

class Mass_Spring_Damper:
    """ Specify the input of a mass-spring-damper system
        """
    def __init__(self, input_sequence,input_sequence_name):
        self.u = np.vstack(input_sequence)
        self.u_name = input_sequence_name

def force_input(experiment_number):
    mass1 = Mass_Spring_Damper(2*np.cos(4*t_p)+3, '$2cos(4t_p)+3$')
    # EXPERIMENT 1: PARAMETERS FOR CORRECT CONCLUSION
    if experiment_number=='1A':
        mass2 = Mass_Spring_Damper(np.zeros(N), '0')
        mass3 = Mass_Spring_Damper(np.zeros(N), '0')
    
    elif experiment_number=='1B':
        # experiment 1B
        mass2 = Mass_Spring_Damper(2*np.cos(4*t_p+3)+3,'$2cos(4t_p+3)+3$')
        mass3 = Mass_Spring_Damper(np.sin(2*t_p),'$sin(2t_p)$')
    
    elif experiment_number=='1C':
        # experiment 1C
        mass2 = Mass_Spring_Damper(2*np.cos(4*t_p+3)+3,'$2cos(4t_p+3)+3$')
        mass3 = Mass_Spring_Damper(np.sin(2*t_p),'$sin(2t_p)$')
    
    # EXPERIMENT 2: PARAMETERS FOR INCORRECT CONCLUSION
    elif experiment_number=='2A':
        # experiment 2A
        mass2 = Mass_Spring_Damper(2*np.cos(4*t_p)+3,'$2cos(4t_p)+3$')
        mass3 = Mass_Spring_Damper(np.sin(2*t_p),'$sin(2t_p)$')
    
    elif experiment_number=='2B':
        # experiment 2B
        mass2 = Mass_Spring_Damper(2*np.cos(4*t_p)+3,'$2cos(4t_p)+3$')
        mass3 = Mass_Spring_Damper(np.sin(2*t_p),'$sin(2t_p)$')
    
    elif experiment_number=='2C':
        # experiment 2C
        mass2 = Mass_Spring_Damper(2*np.cos(4*t_p+2)+3,'$2cos(4t_p+2)+3$')
        mass3 = Mass_Spring_Damper(np.sin(2*t_p),'$sin(2t_p)$')
    
    return mass1, mass2, mass3

def folders(experiment_number):
    '''Define folder to get data from and save data to based on type of test to run (full has 1000 trials, test has 10 trials)'''
    if experiment_number=='1A' or experiment_number=='1B' or experiment_number=='1C':
        folder_figures = 'correct-conclusions/figures_em/'
        while True:
            try:
                test_type = input("Enter the type of test you want to run 'full' or 'test':")
                if test_type== 'full':
                    trials = 1000
                    folder = 'correct-conclusions/exp_'
                    break
                elif test_type== 'test':
                    trials = 10
                    folder = 'test_data/exp_'
                    break
            except ValueError: print("Error!")
    elif experiment_number=='2A' or experiment_number=='2B' or experiment_number=='2C':
        folder_figures = 'incorrect-conclusions/figures_em/'
        while True:
            try:
                test_type = input("Enter the type of test you want to run 'full' or 'test':")
                if test_type== 'full':
                    trials = 1000
                    folder = 'incorrect-conclusions/exp_'
                    break
                elif test_type== 'test':
                    trials = 10
                    folder = 'test_data/exp_'
                    break
            except ValueError: print("Error!")
    folder += experiment_number+'/'
    return folder, folder_figures, trials

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
from time_param import h,T,N,t_p,delta_N,steps,N_t

n_m = 3
print('Choose the parameters of the state-space, where experiment 1 makes a data that should lead to correct conclusions and experiment 2 makes a data that should lead to incorrect conclusions.')
while True:
    try:
        experiment_number = input("Enter the number of the experiment (1A, 1B, 1C, 2A, 2B or 2C) and press enter:")
        mass1, mass2, mass3 = force_input(experiment_number)
        break
    except ValueError:
        print("Error!")
folder, folder_figures, trials = folders(experiment_number)
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
n_k = int((n_n/2)+1)    # number of clusters
n_x = 2                 # number of dimensions
# mean, mu
mu = np.zeros((n_k,n_x))
mu[:,0] = [1.0,0.0,0.5,0.5]         # k means for b_{ij}
mu[:,1] = [0.1,0.0,1.0,1.5]         # k means for error in dx
mu_init = mu
# covariance matrix (C_{i,j} = sigma(x_i,x_j))
cov = np.zeros((n_k,n_x,n_x))                       # cov. matrices for each cluster
cov[0,:,:] = [[10**(-3), 0], [0, 10**(-3)]]         # cov. matrix agency
cov[1,:,:] = [[10**(-3), 0], [0, 10**(-3)]]         # cov. matrix no-agency (static)
cov[2,:,:] = [[10**(-1), 0], [0, 10**(-2)]]         # cov. matrix no-agency (dynamic)
cov[3,:,:] = [[10**(-1), 0], [0, 10**(-2)]]         # cov. matrix no-agency (dynamic)
#cov[4,:,:] = [[10**(-1), 0], [0, 10**(-2)]]         # cov. matrix no-agency (dynamic)
cov_init = cov
# mixture proportions of the clusters (i.e. the prior), pi
pi = np.zeros((n_k,n_n))
pi[0,:] = 1/6         # prior of agent
pi[1,:] = 2/6         # prior of static env.
pi[2,:] = 2/6         # prior of dynamic env.
pi[3,:] = 1/6         # prior of dynamic env.
#pi[4,:] = 1/6         # prior of dynamic env.
#pi[:,1] = [1,0,0,0,0]
pi_init = pi

#--Initialize matrices to store data for all time steps
data = np.zeros((N,n_n,n_x))            # observations, X
r_i = np.zeros((n_k,N,n_n,trials))      # sense of agency (responsibility)
x_hat = np.zeros(shape=x.shape)         # prediction of states
x_hat[0,:,:] = x0                       # initial condition x
y_hat = np.zeros(shape=y.shape)         # prediction of observations
dx_hat = np.zeros(shape=dx.shape)       # prediction of motion states
dx_hat[0,:,:] = dx[0,:,:]               # initial condition dx
eps_x = np.zeros(shape=x.shape)         # prediction error in states
eps_y = np.zeros(shape=y.shape)         # prediction error in observations
eps_dx = np.zeros(shape=dx.shape)       # prediction error in motion states
B_hat =  np.zeros(shape=dx.shape)       # agency estimation (EM and prior adjusted)

# stores parameters of agency clusters
mu_agency = np.zeros((N,n_x,trials))
pi_agency = np.zeros((N,n_n,trials))
cov_agency = np.zeros((N,trials))
t_agency = np.zeros(2*n_m)          # time of agency per state
i_agency = np.zeros((N,n_n,trials))
soa = np.zeros((N,n_n,trials))      # save sense of agency
eps_dx_agency = 0                   # cumulative error of the agent state

# pre-calculations for forward euler step
I = np.identity(A.shape[0])
Ad = I + h*A

# iteration specifications of EM
max_iter = 5
tol = 0.001
steps_window = delta_N

#--Perform EM for each trial
for t in range(4):
    #print('Trial:',t)
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
        eps_dx[i,:,t] = (np.abs(dx_hat[i-delta_N:i+1,:,t]-dx[i-delta_N:i+1,:,t])).mean(axis=0)
        #eps_y[i,:,t] = (np.abs(y_hat[i-delta_N:i+1,:,t]-y[i-delta_N:i+1,:,t])).mean(axis=0)
        
        # save the data at this time step in data (i.e. X)
        data[i,:,0] = B_i[i,:,t]
        data[i,:,1] = eps_dx[i,:,t]
        
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
        
        # postdictive process of finding SoA (w prediction error and causality)
        if i>10:
            i_k = np.arange(0,n_k)                  # range of all possible clusters
            mask = np.ones(len(i_k), dtype=bool)    # boolean mask
            static = np.where(mu[:,0]<0.1)[0]       # static cluster(s)
            mask[static] = False                    # no-agency cluster
            i_agent = i_k[mask]                     # remove from agency options
            # while more than one possible agency cluster
            while sum(mask)>1:
                max_cov = i_agent[np.argmax(cov[i_agent,0,0])]  # maximum covariance
                mask[max_cov] = False   # no-agency cluster
                i_agent = i_k[mask]     # remove max cov from agency k
                if sum(mask)==1: break
                max_eps = i_agent[np.argmax(mu[i_agent,1])] # maximum pred error
                mask[max_eps] = False   # no-agency cluster
                i_agent = i_k[mask]     # remove max error from agency k
            i_agent = i_agent[0]        # choose integer
            i_env = i_k[~mask]          # no-agency clusters are opposite of agency cluster

                # agency parameter, B_hat, with mu and r_i for x_hat and dx_hat
                B_hat[i,:,t] = np.dot(pi[[i_agent],:].T,mu[i_agent,[0]])
            
                # prediction of next step with forward euler (except for last loop)
                if i!=(N-1):
Bd = h*B_hat[i,:,t]
x_hat[i+1,:,t] = Ad.dot(x[i,:,t]) + Bd*mass1.u[i,0]
dx_hat[i+1,:,t] = A.dot(x_hat[i+1,:,t]) + B_hat[i,:,t].dot(mass1.u[i+1,0])
    #agent_dx[i+1,:,:] = B_hat[i,:,t].dot(mass1.u[i+1,0])
    #spring_dx[i,:,:] = dx[i,:,:] - A.dot(x[i,:,:])
    
    # store the agency data for each time step
    t_agency += h*r_i[i_agent,i,:,t]     # time that SoA
    t_agent_1trial += h*r_i[i_agent,i,:,t]
    mu_agency[i,:,t] = mu[i_agent,:]
    cov_agency[i,t] = cov[i_agent,0,0]
    pi_agency[i,:,t] = pi[i_agent,:]
    soa[i,:,t] = r_i[i_agent,i,:,t]
    i_agency[i,:,t] = i_agent
    eps_dx_agency += np.sum(np.abs(eps_dx[i,i_agent,t]))
    
    print('Time agent after trial',t,':',t_agent_1trial)

#--SAVE DATA FOR COMPARISON
print('Saving data...')
mu_agency_mean = np.mean(mu_agency[:,0,:])
cov_agency_mean = np.mean(cov_agency)
pi_agency_mean = np.mean(pi_agency[:,1,:])
B_hat_mean = np.mean(B_hat,axis=(0,2))
B_hat_var = np.var(B_hat,axis=(0,2))
eps_dx_mean = np.mean(eps_dx,axis=(0,2))
eps_dx_var = np.var(eps_dx,axis=(0,2))
np.savetxt(folder+'B_hat_em.txt',B_hat.reshape(-1,B_hat.shape[2]))
np.savetxt(folder+'data_em_'+experiment_number+'.txt',(B_hat_mean,eps_dx_mean,B_hat_var,eps_dx_var,t_agency))
np.savetxt(folder+'data_em-2_'+experiment_number+'.txt',(mu_agency_mean,cov_agency_mean,pi_agency_mean))

#-- SAVE VALUES FOR ANIMATION
'''
    np.savetxt(folder+'em_predictions/em_B_hat.txt',B_hat[:,:,0])
    np.savetxt(folder+'em_predictions/em_dx_hat.txt',dx_hat[:,:,0])
    np.savetxt(folder+'em_predictions/em_x_hat.txt',x_hat[:,:,0])
    np.savetxt(folder+'em_predictions/em_rik_agent.txt', r_i[i_agent,:,:]*100, fmt='%1.2f')
    np.savetxt(folder+'em_predictions/em_pi_agency.txt', pi_agency, fmt='%1.2f')
    np.savetxt(folder+'em_predictions/em_dx_agent.txt',agent_dx[:,:,0])
    np.savetxt(folder+'em_predictions/em_dx_spring.txt',spring_dx[:,:,0])
    '''
#--PLOT
print('Saving figures to:',folder_figures)
# ajdust cluster names according to actual nr of cluster
u_name = [mass1.u_name, mass2.u_name, mass3.u_name]
clusters = list(range(n_n+n_k))
clusters_init = list(range(n_n+n_k))
clusters[n_n+i_agent] = str(i_agent)+' (agency)'
clusters_init[n_n+0] = str(0)+' (agency)'
for i in range(n_n):
    clusters[i] = '$j=%s$'%(i)
    clusters_init[i] = '$j=%s$'%(i)
for i in range(len(i_env)):
    clusters[n_n+i_env[i]] = str(i_env[i])+' (no agency)'
    clusters_init[n_n+i+1] = str(i+1)+' (no agency)'
k_clusters = np.arange(n_k)
k_clusters[i_agent] = 0
k_clusters[i_env] = np.arange(1,n_k)
color_b = ['C1','C0','C7','C3','C4','C5']

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

#--FIGURE 1: plot the responsibility r_i
fig1 = plt.figure(1, figsize=(11, 7))
plt.subplots_adjust(hspace=0.6,wspace=0.3,top=0.95,left=0.1,right=0.9) # spacing
for i in range(n_k):
    # Mass 1: compare real solution of dx (without noise) to dx calculated from LSQ B
    ax1 = plt.subplot(3,2,i+1)
    plt.title('$P(Z_{ij}=%s|b_{j})$ over time for trial %s'%(k_clusters[i],trial))
    plt.ylabel('$r_i(Z_{ij}=k)$')
    plt.xlabel('Time /s')
    plt.scatter(t_p,r_i[i,:,0,trial],s=5,label='$P(Z_{ij}=k|b_{1})$',color=color_b[0])
    plt.scatter(t_p,r_i[i,:,1,trial],s=5,label='$P(Z_{ij}=k|b_{2})$',color=color_b[1])
    plt.scatter(t_p,r_i[i,:,2,trial],s=5,label='$P(Z_{ij}=k|b_{3})$',color=color_b[2])
    plt.scatter(t_p,r_i[i,:,3,trial],s=5,label='$P(Z_{ij}=k|b_{4})$',color=color_b[3])
    plt.scatter(t_p,r_i[i,:,4,trial],s=5,label='$P(Z_{ij}=k|b_{5})$',color=color_b[4])
    plt.scatter(t_p,r_i[i,:,5,trial],s=5,label='$P(Z_{ij}=k|b_{6})$',color=color_b[5])
# Position legend
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width, box.height])   # set box pos.
ax1.legend(loc='center left', bbox_to_anchor=(1.3, 0.4))    # put a legend to the right of the current axis

fig1.set_rasterized(True)
plt.savefig(folder_figures+'exp'+experiment_number+'_agent_rik.eps', format='eps')

#--FIGURE 2: plot the b values
fig2 = plt.figure(2, figsize=(7.5, 5))
plt.subplots_adjust(hspace=1.25,top=0.85,left=0.19,right=0.90) # spacing
fig2.suptitle('$\mathbf{\hat B}$ over time for trial %s' %(trial))
for i in range(0,n_m):
    ax1 = plt.subplot(n_m,1,i+1)
    plt.title('Mass $m_%s$ with force input %s'%(i+1,u_name[i]), y=0.98)
    plt.xlabel('Time /s')
    plt.ylabel('$b_{ij}$')
    plt.grid()
    plt.ylim(np.min((min(B_hat[:,2*i,trial]),min(B_hat[:,2*i+1,trial]),min(B_a[2*i]),min(B_a[2*i+1])))-0.075,np.max((max(B_hat[:,2*i,trial]),max(B_hat[:,2*i+1,trial]),max(B_a[2*i]),max(B_a[2*i+1])))+0.075)
    # plot real B values
    plt.axhline(B_a[2*i], color='k', linestyle=':', label='$b_{%s}$ (true)'%(2*i))
    plt.axhline(B_a[2*i+1], color='k', linestyle='--', label='$b_{%s}$ (true)'%(2*i+1))
    # plot OLS estimated B_hat
    #plt.scatter(t_p, ols_B_hat[:,2*i,trial], s=4, color='C6',label='$\hatb_{%s}$ (OLS)'%(2*i))
    #plt.scatter(t_p, ols_B_hat[:,2*i+1,trial], s=4, color='C1',label='$\hatb_{%s}$ (OLS)'%(2*i+1))
    # plot EM estimated B_hat
    plt.scatter(t_p, B_hat[:,2*i,trial], s=4, color=color_b[2*i], label='$\hatb_{i%s}$ (EM)'%(2*i))
    plt.scatter(t_p, B_hat[:,2*i+1,trial], s=4, color=color_b[2*i+1], label='$\hatb_{i%s}$ (EM)'%(2*i+1))
    
    # Position legend
    box = ax1.get_position()
    ax1.set_position([box.x0-box.width * 0.07, box.y0, box.width * 0.9, box.height]) # move box to left by 5% and shrink current axis by 10% (i.e. center box)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))        # put a legend to the right of the current axis
fig2.set_rasterized(True)
plt.savefig(folder_figures+'exp'+experiment_number+'_agent_b.eps', format='eps')

#--FIGURE 3: Plot contours of clusters after EM
x1 = np.linspace(-1,2.5,2000)
x2 = np.linspace(-1,2,1500)
X, Y = np.meshgrid(x1,x2)
pos = np.empty(X.shape + (2,)) # a new array of given shape and type, without initializing entries
pos[:, :, 0] = X; pos[:, :, 1] = Y

fig3,ax = plt.subplots(figsize=(7, 6))
plt.subplots_adjust(top=0.94,right=0.96,left=0.125,bottom=0.12) # spacing
plt.title('Clusters after point mass mRHI at $T=20$s')
# plot B estimates
Z = dict()
CS = dict()
SC = dict()
lines = []
labels = clusters
color_k = ['C2','C1','C8','C6','C9']
color_k[0] = color_k[i_agent]
color_k[i_agent] = 'C2'

for i in range(0,2*n_m):
    SC[i] = plt.scatter(data[N_t-delta_N:N_t,i,0], data[N_t-delta_N:N_t,i,1],c=color_b[i], s=10)
    lines.append(SC[i])
for i in range(0,n_k):
    Z[i] = stats.multivariate_normal(mu[i,:], cov[i,:,:])
    CS[i] = plt.contour(X, Y, Z[i].pdf(pos), levels=5, colors=color_k[i], alpha=0.8)
    lines.append(CS[i].collections[-1])
plt.xlabel('$b_{ij}$')
plt.ylabel('$\epsilon_{\dot{x}_{ij}}$')
plt.xlim(np.min(data[N_t-delta_N:N_t,:,0])-0.2, np.max(data[N_t-delta_N:N_t,:,0])+0.05)
plt.ylim(np.min(data[N_t-delta_N:N_t,:,1])-0.1, np.max(data[N_t-delta_N:N_t,:,1])+0.5)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.8])    # set box pos.
ax.legend(lines,clusters_init,loc='lower center', ncol=4, bbox_to_anchor=(0.45, -0.425))    # put a legend to the right of the current axis


fig3.set_rasterized(True)
plt.savefig(folder_figures+'exp'+experiment_number+'_agent_k_after.eps', format='eps')

#--FIGURE 4: Plot of contours of clusters before EM
fig4, ax = plt.subplots(figsize=(7, 6))
plt.subplots_adjust(top=0.94,right=0.96,left=0.125,bottom=0.12) # spacing
plt.title('Clusters before point mass mRHI at $T=0$s')
color_init = ['C2','C1','C8','C6','C9']
lines = []
for i in range(0,2*n_m):
    SC[i] = plt.scatter(data[0:delta_N,i,0],data[0:delta_N,i,1],c=color_b[i],s=10)
    lines.append(SC[i])
for i in range(0,n_k):
    Z[i] = stats.multivariate_normal(mu_init[i,:], cov_init[i,:])
    CS[i] = plt.contour(X, Y, Z[i].pdf(pos), levels=5, colors=color_init[i], alpha=0.8)
    lines.append(CS[i].collections[0])
plt.xlabel('$b_{ij}$')
plt.ylabel('$\epsilon_{\dot{x}_{ij}}$')
plt.xlim(np.min(data[0:delta_N,:,0])-0.05,np.max(data[0:delta_N,:,0])+0.3)
plt.ylim(np.min(data[0:delta_N,:,1])-0.1,2)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.8])   # set box pos.
ax.legend(lines,clusters_init,loc='lower center', ncol=4, bbox_to_anchor=(0.45, -0.425))    # put a legend to the right of the current axis

fig4.set_rasterized(True)
plt.savefig(folder_figures+'exp'+experiment_number+'_agent_k_before.eps', format='eps')

#--FIGURE 5: Plot the agent's sense of agency
fig5, ax = plt.subplots(figsize=(7, 4))
plt.subplots_adjust(top=0.9,left=0.15,right=0.8,bottom=0.14) # spacing
plt.title('The agent\'s SoA ($Z_{ij}=%s$) over time for trial %s'%(i_agent,trial))
plt.ylabel('$r_i(Z_{ij}=%s)$'%(i_agent))
plt.xlabel('Time /s')
plt.scatter(t_p,soa[:,0,trial],s=5,label='$P(Z_{ij}=k|j=0)$',color=color_b[0])
plt.scatter(t_p,soa[:,1,trial],s=5,label='$P(Z_{ij}=k|j=1)$',color=color_b[1])
plt.scatter(t_p,soa[:,2,trial],s=5,label='$P(Z_{ij}=k|j=2)$',color=color_b[2])
plt.scatter(t_p,soa[:,3,trial],s=5,label='$P(Z_{ij}=k|j=3)$',color=color_b[3])
plt.scatter(t_p,soa[:,4,trial],s=5,label='$P(Z_{ij}=k|j=4)$',color=color_b[4])
plt.scatter(t_p,soa[:,5,trial],s=5,label='$P(Z_{ij}=k|j=5)$',color=color_b[5])
box = ax.get_position()
ax.set_position([box.x0-box.width * 0.07, box.y0, box.width * 0.9, box.height])   # set box pos.
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))    # put a legend to the right of the current axis

fig5.set_rasterized(True)
plt.savefig(folder_figures+'exp'+experiment_number+'_agent_SoA.eps', format='eps')

#--FIGURE 7: compare real solution of observed dx to predicted dx calculated from EM B
fig7 = plt.figure(7, figsize=(11, 7))
plt.subplots_adjust(hspace=0.60,top=0.90,left=0.15,right=0.85) # spacing
fig7.suptitle('$\mathbf{\dot{x}}$ over time for trial %s' %(trial))
for i in range(n_m):
    ax1 = plt.subplot(3,1,i+1)
    plt.title('Mass $m_%s$ with force input %s'%(i+1,u_name[i]), y=0.98)
    plt.ylabel('$\dot{\mathbf{x}}$')
    plt.xlabel('Time /s')
    plt.grid(alpha=0.5)
    
    # plot observed dx
    ax1.scatter(t_p, dx[:,2*i,trial],s=1,label='Observed vel. ($\dot{x}_{i%s}$)'%(2*i))
    ax1.scatter(t_p, dx[:,2*i+1,trial],s=1,label='Observed acc. ($\dot{x}_{i%s}$)'%(2*i+1))
    # plot predicted dx calculated from prior
    ax1.scatter(t_p, dx_hat[:,2*i,trial],s=3,label='Predicted vel. ($\dot{\hatx}_{i%s}$)'%(2*i))
    ax1.scatter(t_p, dx_hat[:,2*i+1,trial],s=3,label='Predicted acc. ($\dot{\hatx}_{i%s+1}$)'%(2*i+1))
    # plot error of sum of dx1 and dx2
    ax1.plot(t_p, eps_dx[:,2*i,trial],'--',color='k',label='$\epsilon_{\dot{x}_{i%s}}+\epsilon_{\dot{x}_{i%s}}$'%(2*i,2*i+1))
    
    # Position legend
    box = ax1.get_position()
    ax1.set_position([box.x0-box.width * 0.07, box.y0, box.width * 0.9, box.height]) # move box to left by 5% and shrink current axis by 10% (i.e. center box)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))        # put a legend to the right of the current axis
fig7.set_rasterized(True)
plt.savefig(folder_figures+'exp'+experiment_number+'_agent_dx.eps', format='eps')

#--FIGURE 9: plot b_2 over time
fig9, ax1 = plt.subplots(figsize=(7.5, 5))
plt.subplots_adjust(top=0.9,bottom=0.15,left=0.15,right=0.95) # spacing
fig9.suptitle('$\hat{b}_2$ over time for trial %s' %(trial))
plt.xlabel('Time /s')
plt.ylabel('$b_j$')
plt.grid()
plt.ylim(B_a[1]-0.05,B_a[1]+0.05)
# plot real b_1 values
plt.axhline(B_a[1], color='k', linestyle=':', label='$b_{2}$ (true)')
# plot OLS B_hat
plt.scatter(t_p, ols_B_hat[:,1,trial], s=5, color='C1', label='$\hatb_{2}$ (OLS)')
#plt.scatter(t_p, mu_agency[:,0,trial], s=5, color='C2', label='Agency $\mu_%s$'%(i_agent))
# plot EM B_hat
plt.scatter(t_p, B_hat[:,1,trial], s=5, color='C0', marker='+' ,label='$\hatb_{i2}$ (EM)')
# plot mean of agency, \mu
plt.legend()
# save image
fig9.set_rasterized(True)
plt.savefig(folder_figures+'exp'+experiment_number+'_agent_b2.eps', format='eps')

#--FIGURE 10: plot variance over time
fig10, ax1 = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(top=0.9,bottom=0.15,left=0.15,right=0.95) # spacing
fig10.suptitle('Variance of agency cluster over time for trial %s' %(trial))
plt.xlabel('Time /s')
plt.ylabel('$\sigma^2$')
plt.grid()
#plt.ylim(np.min((min(B_hat[:,1,trial]), min(B_a[1])))-0.05, np.max((max(B_hat[:,1,trial]),max(B_a[1])))+0.05)
# plot OLS B_hat
plt.scatter(t_p, cov_agency[:,trial], s=5, color='C0',label='$\sigma^2$ in $b_{ij}$')
plt.legend()

# save image
fig10.set_rasterized(True)
plt.savefig(folder_figures+'exp'+experiment_number+'_agent_var.eps', format='eps')

#-- PRINT: details
print('\n-------------------------------------------------------------------')
print('Performance of experiment',experiment_number)

print('\nMEAN OF:')
print('\tincremental B:',np.mean(B_i,axis=(0,2)))
print('\tOLS B_hat:', np.mean(ols_B_hat,axis=(0,2)))
print('\tEM B_hat',B_hat_mean)
print('\tmean of agency mu:',np.mean(mu_agency[:,0,:]))

print('\nVARIANCE OF:')
print('\tincremental B:', np.var(B_i,axis=(0,2)))
print('\tOLS B_hat:', np.var(ols_B_hat,axis=(0,2)))
print('\tEM B_hat:', B_hat_var)
print('\tmean of agency covariance:', np.mean(cov_agency))

print('\nIN FINAL TIME WINDOW,')
print('mean:')
print('\tOLS B_hat:', np.mean(ols_B_hat[-delta_N:,:,:],axis=(0,2)))
print('\tEM B_hat',np.mean(B_hat[-delta_N:,:,:],axis=(0,2)))
print('\tmean of agency mu:',np.mean(mu_agency[-delta_N:,0,:]))
print('variance:')
print('\tOLS B_hat:', np.var(ols_B_hat[-delta_N:,:,:],axis=(0,2)))
print('\tEM B_hat:', np.var(B_hat[-delta_N:,:,:],axis=(0,2)))
print('\tmean of agency covariance:', np.mean(cov_agency[-delta_N:,:]))

print('\nIn average reponsibility the probability that')
print('\tof agency over state:',np.mean(r_i[i_agent,:,:,:],axis=(0,2)))
print('\tof no-agency 1 over state:',np.mean(r_i[i_env[0],:,:,:],axis=(0,2)))
print('\tof no-agency 2 over state:',np.mean(r_i[i_env[1],:,:,:],axis=(0,2)))
print('\tof no-agency 3 over state:',np.mean(r_i[i_env[2],:,:,:],axis=(0,2)))

print('AGENCY ERROR')
print('cumulative:', eps_dx_agency)
print('mean:',eps_dx_agency/(N*trials))

print('\nClusters for: agency, k=',i_agent, 'no-agency, k=',i_env)
print('\nThe mixture proportions are:')
print(pi)

print('\nPERCENTAGE TIME')
print('\tof agency per state:',100*(t_agency/(T*trials)),'%')


plt.show() # uncomment if you want to plot



