"""
Simulate Basal Ganglia - Go/NoGo mechanism
"""
import numpy as np
from matplotlib import pyplot as plt
from multi_armed_bandits import BernoulliBandit
import time

def sigmoid(x, a = 1, x0 = 0):
    x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
    #x = 1 / (1 + np.exp(a*x))
    return x

def define_synapses(Ns):
    """
        Definition of synapses involved
        -------------------------------

        Ns: number of stimuli
        Wit: gpi to thalamus (inh)
        Wne: NoGO to gpe (inh)
        Wgi: GO to gpi (inh)

        STN based:
        ----------

        W_e_STN: gpe to STN (inh)
        W_STN_e: STN to gpe
        W_STN_i: STN to gpi

        Synpases between them:
        ---------------------

        W_STN: within STN
        W_e: within gpe

        Note: Except for W_STN everything are diagonal (i.e the synapse connection is one to one (i=j))
    """
    inhb = -1    # inhibtion multiplier

    Wit = np.zeros(shape=(Ns,Ns))
    Wne = np.zeros(shape=(Ns,Ns))
    Wgi = np.zeros(shape=(Ns,Ns))

    W_e_STN = np.zeros(shape =(Ns,Ns))
    W_STN_e = np.zeros(shape =(Ns,Ns))
    W_STN_i = np.zeros(shape =(Ns,Ns))
    
    W_STN = np.random.random(size = (Ns, Ns))
    W_e = np.random.random(size = (Ns, Ns))

    np.fill_diagonal(Wit, inhb * np.random.random())
    np.fill_diagonal(Wne, inhb * np.random.random()) 
    np.fill_diagonal(Wgi, inhb * np.random.random())
    
    np.fill_diagonal(W_e_STN, inhb * np.random.random())
    np.fill_diagonal(W_STN_e, np.random.random())
    np.fill_diagonal(W_STN_i, np.random.random()) 
    
    np.fill_diagonal(W_e, 0)    

    return(Wit, Wne, Wgi, W_e_STN, W_STN_e, W_STN_i, W_STN, W_e)

def calculate_reward_for_actions(winning_slots, reward_vector):
    
    reward_score = 0
    for i in winning_slots[0]:
        if reward_vector[i] == 1:
            reward_score += 1
    return reward_score

def define_parameters(Ns, simulation_length):
    """
        Initialize structural parameters of Basal Ganglia
        -------------------------------------------------

        GO (D1 receptors):
        --
        act_Go: activation
        
        NoGO (D2 receptors):
        ----
        act_NGo: activation
        
        Globus pallidus externa:
        ------------------------
        act_GPe: activation
        
        Globus pallidus interna:
        -----------------------
        act_GPi: activation
        
        Thalamus:
        --------
        act_Th: activation
        
        Sub-Thalamic Nucleus:
        --------------------
        act_STN: activation
    """

    act_Go = np.zeros(shape =(simulation_length, Ns))
    act_NGo = np.zeros(shape =(simulation_length, Ns))
    act_GPe = np.zeros(shape =(simulation_length, Ns))
    act_GPi = np.zeros(shape =(simulation_length, Ns))
    act_Th = np.zeros(shape =(simulation_length, Ns))
    act_STN = np.zeros(shape =(simulation_length, Ns))
    
    return (act_Go, act_NGo, act_GPe, act_GPi, act_Th, act_STN)

def define_gains():
    """
    Gains:
        -----
        alpha: DA to GO (exh)
        beta: DA to NoGO (inh)
    """
    alpha = 0.75
    beta = -1
    return alpha, beta

def BG(Stimulus, bandit, W_g, W_n, dopamine_tonic):
    """
        Learning Dynamics
        ----------------

        Time pattern of phasic dopamine:
        ------------------------------
        latency
        duration

        Learnable synapses:
        ------------------
        W_g: synapses connecting Stimulus to GO (D1 receptor)
        W_n: synapses connecting Stimulus to NoGO (D2 receptor)
        shape: (Ns, Ns), becuase each neuron in D1 have one connection with all available actions in Stimulus
    """
    Ns = Stimulus.shape[0]   # number of stimulus
    
    simulation_length = 50
    
    temporal_diff_err = np.zeros((simulation_length, Ns))
    gradient_clipp = np.zeros((simulation_length, Ns))
    delta_Stimulus = np.zeros((simulation_length, Ns))
    reward_awarded = np.zeros(simulation_length)
    tot_reward_awarded = 0
    winning_slots = []

    act_Go, act_NGo, act_GPe, act_GPi, act_Th, act_STN = define_parameters(Ns, simulation_length)

    Wit, Wne, Wgi, W_e_STN, W_STN_e, W_STN_i, W_STN, W_e = define_synapses(Ns)

    alpha, beta = define_gains()

    temp = []
    rew = []

    for t in range(simulation_length - 1):

        reward_vector = []
        for i in range(bandit.n):
            reward_vector.append(bandit.get_reward(i))

        max_possible_reward = np.sum(reward_vector)

        # find sigmoid activations
        act_Go[t,:] = sigmoid(np.dot(Stimulus, W_g))
        act_NGo[t,:] = sigmoid(np.dot(Stimulus, W_n))
        
        act_GPe[t,:] = sigmoid(np.dot(act_NGo[t,:], Wne) + np.dot(act_STN[t,:], W_STN_e))
        act_STN[t,:] = sigmoid(np.dot(act_GPe[t,:], W_e_STN))
        act_GPi[t,:] = sigmoid(np.dot(act_Go[t,:], Wgi) + np.dot(act_STN[t,:], W_STN_i))
        
        act_Th[t,:] = sigmoid(np.dot(act_GPi[t,:], Wit))
        
        # TODO: use W_STN and W_e to moderate W_STN_e and W_e_STN

        winning_threshold = 0.5

        winning_slots = np.where(act_Th[t,:] > winning_threshold)

        temporal_diff_err[t,:] = reward_vector - act_Th[t,:]
        temporal_diff_err[t,:][temporal_diff_err[t,:] < 0] = 0

        reward_awarded[t] = calculate_reward_for_actions(winning_slots, reward_vector)
        tot_reward_awarded += reward_awarded[t]

        if t == 0:
            gradient_clipp[t,:] = 0
        else:
            gradient_clipp[t,:] = act_Th[t] - act_Th[t-1]

        lr = 10/(t+1)

        delta_Wn = lr * temporal_diff_err[t,:] * Stimulus
        delta_Wg = lr * temporal_diff_err[t,:] * Stimulus
        
        W_n = W_n - delta_Wn
        W_g = W_g - delta_Wg

        D_hi = 0.1
        D_low = -0.3

        if t > 0:
            for sim in range(Ns):
                if gradient_clipp[t,sim] > D_hi:
                    delta_Stimulus[t + 1,sim] = delta_Stimulus[t,sim]
                
                elif gradient_clipp[t,sim] < D_hi and gradient_clipp[t,sim] >= D_low:
                    delta_Stimulus[t+1,sim] = delta_Stimulus[t,sim] + np.random.random()
                
                elif gradient_clipp[t,sim] < D_low:
                    delta_Stimulus[t+1,sim] = -1 * delta_Stimulus[t,sim]

        Stimulus = Stimulus + delta_Stimulus[t+1,:]
        temp.append(np.sum(temporal_diff_err[t,:]))
        rew.append(reward_awarded[t]/max_possible_reward)

    plt.plot(temp, label = "Temp_error")
    plt.xlabel("Epochs")
    plt.ylabel("Temp_err")
    plt.title("Temp_Err vs Epochs")
    plt.legend()
    plt.show()
    
    plt.plot(rew, label = "Rewards")
    plt.xlabel("Epochs")
    plt.ylabel("Rewards")
    plt.title("Rewards vs Epochs")
    plt.legend()
    plt.show()
    
    return Stimulus, temporal_diff_err[simulation_length - 2,:], gradient_clipp[simulation_length - 2,:], winning_slots[len(winning_slots) - 1], W_g, W_n

if __name__  == "__main__":
    
    bandit = BernoulliBandit()
    estimate_prob = 0.7 
    #Stimulus = bandit.prob_dist * estimate_prob
    Stimulus = np.random.random(bandit.n) # or to have random start
    W_g = np.random.random((Stimulus.shape[0], Stimulus.shape[0]))
    W_n = np.random.random((Stimulus.shape[0], Stimulus.shape[0]))
    dopamine_tonic = 1.4
    
    Stimulus, temporal_diff_err, gradient_clipp, winning_slots, W_g, W_n = BG(Stimulus, bandit, W_g, W_n, dopamine_tonic)