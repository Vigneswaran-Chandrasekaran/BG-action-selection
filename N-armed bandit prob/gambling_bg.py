"""
Simulate Basal Ganglia - Go/NoGo mechanism for N-armed bandits
"""
import numpy as np
from matplotlib import pyplot as plt
from multi_armed_bandits import BernoulliBandit
import time

def sigmoid(x, dopamine_flag = False, a = 4, sign = 'n', norm_only = False):

    # normalize the values to avoid saturation
    x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
    
    # do only normalization
    if norm_only == True:
        return x
    
    # check dopamine should be added
    if dopamine_flag == True:
        alpha, beta = define_gains()
        if sign == 'n':
            x = 1 / (1 + np.exp(-1 * a * beta * x))
        elif sign == 'p':
            x = 1 / (1 + np.exp(-1 * a * alpha * x))
        else:
            raise NameError("Undefined string const passed")
        return x    
   
    # if no dopamine is added (dopamine_flag = False)
    x = 1 / (1 + np.exp(-1 * a * x))
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
    if reward_vector[winning_slots] == 1:
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
    alpha = 1.75
    beta = -1
    return alpha, beta

def BG(Stimulus, bandit, W_g, W_n):
    """
        Learnable synapses:
        ------------------
        W_g: synapses connecting Stimulus to GO (D1 receptor)
        W_n: synapses connecting Stimulus to NoGO (D2 receptor)
        shape: (Ns, Ns), becuase each neuron in D1 have one connection with all available actions in Stimulus
    """
    # number of stimulus (i.e. number of arms/slots)
    Ns = Stimulus.shape[0]   
    # number of episodes for agent
    simulation_length = 1000
    # temporal difference error [reward - action value]
    temporal_diff_err = np.zeros((simulation_length, Ns))
    # gradient step [gradient(t) - gradient(t-1)]
    gradient_clipp = np.zeros((simulation_length, Ns))
    # step change [x(t) - x(t-1)]
    delta_Stimulus = np.zeros((simulation_length, Ns))
    # reward awarded for choosing the action_i/arm_i/slot_i
    reward_awarded = np.zeros(simulation_length)
    # cumulative reward awarded so far
    tot_reward_awarded = 0

    # define activation monitor for various neurons and components
    act_Go, act_NGo, act_GPe, act_GPi, act_Th, act_STN = define_parameters(Ns, simulation_length)
    # define various synapses [non-learnable] connecting various components
    Wit, Wne, Wgi, W_e_STN, W_STN_e, W_STN_i, W_STN, W_e = define_synapses(Ns)
    # lists to monitor performance
    temp = []
    rew = []
    # flag to check whether dopamine influence is on/off
    # TODO: control the magnitude of dopamine using gradient_clip history?
    dopamine_flag = False

    for t in range(simulation_length - 1):
        # vector with 1s and 0s representing corresponding slots reward for particular time instant
        # based on underlying pdf
        reward_vector = []
        for i in range(bandit.n):
            reward_vector.append(bandit.get_reward(i))
        
        # how many slots have rewarded for the particular time instant
        max_possible_reward = np.sum(reward_vector)
        
        # find sigmoidal activations
        # if dopmaine_flag = True, then corresponding gain parameters are added to increase the slope
        # of sigmoid function. If sign = p is set, it produces positive effect and sign = n produces negative
        act_Go[t,:] = sigmoid(np.dot(Stimulus, W_g), dopamine_flag = dopamine_flag, sign = 'p')
        act_NGo[t,:] = sigmoid(np.dot(Stimulus, W_n), dopamine_flag = dopamine_flag, sign = 'n')
        # sigmoid activations not affected by dopamine directly
        act_GPe[t,:] = sigmoid(np.dot(act_NGo[t,:], Wne) + np.dot(act_STN[t,:], W_STN_e))
        act_STN[t,:] = sigmoid(np.dot(act_GPe[t,:], W_e_STN))
        act_GPi[t,:] = sigmoid(np.dot(act_Go[t,:], Wgi) + np.dot(act_STN[t,:], W_STN_i))
        # Thalamus activation [action values]
        act_Th[t,:] = sigmoid(np.dot(act_GPi[t,:], Wit))
        action_values = act_Th[t,:]
        # TODO: use W_STN and W_e to moderate W_STN_e and W_e_STN
        # Highest activation is considered as winner
        winning_slots = np.argmax(action_values)
        # calculate temp diff error
        temporal_diff_err[t,:] = reward_vector - act_Th[t,:]
        # learning rate (decays by time)
        lr = 0.5/(t+1)
        # synapses updated by temp_diff_err
        delta_Wn = lr * temporal_diff_err[t,:] * Stimulus
        delta_Wg = lr * temporal_diff_err[t,:] * Stimulus
        W_n = W_n + delta_Wn
        W_g = W_g + delta_Wg
        # check the reward for the action
        reward_awarded[t] = calculate_reward_for_actions(winning_slots, reward_vector)
        tot_reward_awarded += reward_awarded[t]
        # calculate gradient change
        # note: sigmoid() is used only to normalize [norm_only = True] the gradient values (as magnitude of grad diminishes by time)
        # Normalization changes the magnitude of gradient but it doesn't affect the relative value with other slots
        if t == 0:
            # t-1 doesn't exists
            gradient_clipp[t,:] = 0
        else:
            gradient_clipp[t,:] = sigmoid(act_Th[t] - act_Th[t-1], norm_only = False)
        # define threshold to estimate delta_stimulus
        D_hi = 0.65
        D_low = 0.35
        # iterate the stimulus value for each slot in accordance to gradient_clipp[slot]
        for slot in range(Ns):
            # GO action
            if gradient_clipp[t,slot] > D_hi:
                delta_Stimulus[t + 1,slot] = delta_Stimulus[t,slot]
                # dopamine is set to True: (the next action is predicted to be reward gaining)
                dopamine_flag = True
            # Explore (add gaussian noise) and set dopamine false
            elif gradient_clipp[t,slot] < D_hi and gradient_clipp[t,slot] >= D_low:
                delta_Stimulus[t + 1,slot] = delta_Stimulus[t,slot] + np.random.random()
                dopamine_flag = False
            # NoGO action
            elif gradient_clipp[t,slot] < D_low:
                delta_Stimulus[t + 1,slot] = -1 * delta_Stimulus[t,slot]
                dopamine_flag = False
        # update stimulus [take next step]
        Stimulus = Stimulus + delta_Stimulus[t+1,:]
        # monitors 
        temp.append(np.sum(temporal_diff_err[t,:]))
        rew.append(tot_reward_awarded)

    plt.plot(rew, label = "Rewards")
    plt.xlabel("Epochs")
    plt.ylabel("Rewards")
    plt.title("Rewards vs Epochs")
    plt.legend()
    plt.show()
    return Stimulus, temporal_diff_err[simulation_length - 2,:], gradient_clipp[simulation_length - 2,:], winning_slots, W_g, W_n

if __name__  == "__main__":

    # define bandit with necessary variables
    bandit = BernoulliBandit()
    # initial conditon (estimate_prob defines percentage of knowledge about pdf at starting)
    # if inital knowledge needs to be zero: create random Stimulus
    #estimate_prob = 0.7 
    #Stimulus = bandit.prob_dist * estimate_prob
    Stimulus = np.random.random(bandit.n) # or to have random start
    # Weights connecting D1
    W_g = np.random.random((Stimulus.shape[0], Stimulus.shape[0]))
    # Weights connecting D2
    W_n = np.random.random((Stimulus.shape[0], Stimulus.shape[0]))
    # start learning
    Stimulus, temporal_diff_err, gradient_clipp, winning_slots, W_g, W_n = BG(Stimulus, bandit, W_g, W_n)