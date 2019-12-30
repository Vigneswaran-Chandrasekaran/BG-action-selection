"""
Simulate Basal Ganglia - Go/NoGo mechanism
"""

import numpy as np

tau = 15 
tauL = 5 * tau #lateral inhibition time constant
dt = 0.1
start_time = 0
end_time = 500
simulation_length = np.arange(start_time, end_time, dt).shape[0]

def sigmoid(x, a = 4, x0 = 1):
    return (1 / (1 - np.exp(a * x - x0)))

def define_synapses(Nc, Ns):
    """
        Definition of synapses involved
        -------------------------------

        Ns: number of stimuli
        Wcs: stimulus to cortex 
        Wtc: thalamus to cortex (diag,ext)
        Wct: cortex to thalamus (diag,ext)
        Wit: gpi to thalamus (diag,inh)
        Wne: NoGO to gpe (diag,inh)
        Wei: gpe to gpi (diag,inh)
        Wgi: GO to gpi (diag,inh)
        Wlat: lateral (diag,inh)

        STN based:
        ----------

        W_eng_STN: energy to STN
        W_e_STN: gpe to STN (inh)
        W_STN_e: STN to gpe
        W_STN_i: STN to gpi

        ChI based:
        ---------

        W_Go_ChI: GO to ChI (inh)
        W_NGo_ChI: NoGO to ChI
    """
    Wcs = np.ones(shape =(Nc,Ns))
    Wtc = np.zeros(shape=(Nc,Nc))
    Wct = np.zeros(shape=(Nc,Nc))
    Wit = np.zeros(shape=(Nc,Nc))
    Wne = np.zeros(shape=(Nc,Nc))
    Wei = np.zeros(shape=(Nc,Nc))
    Wgi = np.zeros(shape=(Nc,Nc))
    Wlat =  -1.2 * np.ones(shape=(Nc,Nc))

    W_eng_STN = 7
    W_e_STN = -1
    W_STN_e = 1
    W_STN_i = 30
    W_g_ChI = -1
    W_n_ChI = 1

    np.fill_diagonal(Wtc, 4)
    np.fill_diagonal(Wct, 3)
    np.fill_diagonal(Wit, -3)
    np.fill_diagonal(Wne, -2.2) 
    np.fill_diagonal(Wei, -3) 
    np.fill_diagonal(Wgi, -36)
    np.fill_diagonal(Wlat, 0)

    return(Wcs, Wtc, Wct, Wit, Wne, Wei, Wgi, Wlat, W_eng_STN, W_e_STN, W_STN_e, W_STN_i, W_g_ChI, W_n_ChI)

def define_parameters(Nc, Ns):
    """
        Initialize structural parameters of Basal Ganglia
        -------------------------------------------------

        Cortex:
        -------
        Nc: Number of neurons in cortex
        act_C: activation
        Ip_C: sigmoidal input
        Lat_Inh: lateral inhibition contribution

        GO:
        --
        act_Go: activation
        Ip_Go: sigmoidal input

        NoGO:
        ----
        act_NGo: activation
        Ip_NGo: sigmoidal input

        Globus pallidus externa:
        ------------------------
        act_GPe: activation
        Ip_GPe: sigmoidal input

        Globus pallidus interna:
        -----------------------
        act_GPi: activation
        Ip_GPi: sigmoidal input

        Thalamus:
        --------
        act_Th: activation
        Ip_Th: sigmoidal input

        Sub-Thalamic Nucleus:
        --------------------
        act_STN: activation
        Ip_STN: sigmoidal input

        Cholinergic Inter-Neuron:
        ------------------------
        act_ChI: activation
        Ip_ChI: sigmoidal input

        Energy:
        ------
        Eng: energy input
    """
    act_C = np.zeros(shape =(simulation_length, Nc))
    act_Go = np.zeros(shape =(simulation_length, Nc))
    act_NGo = np.zeros(shape =(simulation_length, Nc))
    act_GPe = np.zeros(shape =(simulation_length, Nc))
    act_GPi = np.zeros(shape =(simulation_length, Nc))
    act_Th = np.zeros(shape =(simulation_length, Nc))
    act_STN = np.zeros(simulation_length)
    act_ChI = np.zeros(simulation_length)

    Ip_C = act_C
    Ip_Go = act_Go
    Ip_NGo = act_NGo
    Ip_GPe = act_GPe
    Ip_GPi = act_GPi
    Ip_Th = act_Th
    Ip_STN = act_STN
    Ip_ChI = act_ChI

    Lat_Inh = np.zeros(shape =(simulation_length, Nc))
    Eng = np.zeros(simulation_length)
    Ip_Go_DA = np.zeros(shape = (simulation_length, Nc))
    Ip_NGo_DA = np.zeros(shape = (simulation_length, Nc))

    return act_C, act_Go, act_NGo, act_GPe, act_GPi, act_Th, act_STN, act_ChI, Ip_C, Ip_Go, Ip_NGo, Ip_GPe, Ip_GPi, Ip_Th, Ip_STN, Ip_ChI, Lat_Inh, Eng, Ip_Go_DA, Ip_NGo_DA

def define_gains():
    """
    Gains:
        -----
        alpha: DA to GO (exh)
        beta: DA to NoGO (inh)
        gamma: DA to ChI (inh)
    """
    alpha = 0.75
    beta = -1
    gamma = -0.5
    return alpha, beta, gamma

def BG(Stimulus, Nc, Wsg, Wsn, Wcg, Wcn, dopamine_tonic, correct_winner):
    """
        Learning Dynamics

        Tonic activity:
        --------------
        Te: gpe
        Ti: gpi
        TChI: ChI

        Time pattern of phasic dopamine:
        ------------------------------
        latency
        duration

        Learnable synapses:
        ------------------

        Wcg: cortex to GO
        Wsg: stimulus to GO
        Wcn: cortex to NoGO
        Wsn: stimulus to NoGO
    """
    Ns = Stimulus.shape[0]   # number of stimulus

    Te = 1
    Ti = 3
    TChI = 1
    latency = int(100/dt)
    duration = int(50/dt)

    act_C, act_Go, act_NGo, act_GPe, act_GPi, act_Th, act_STN,act_ChI, Ip_C, Ip_Go, Ip_NGo, Ip_GPe, Ip_GPi, Ip_Th, Ip_STN, Ip_ChI, Lat_Inh, Eng, Ip_Go_DA, Ip_NGo_DA = define_parameters(Nc, Ns)

    Wcs, Wtc, Wct, Wit, Wne, Wei, Wgi, Wlat, W_eng_STN, W_e_STN, W_STN_e, W_STN_i, W_g_ChI, W_n_ChI = define_synapses(Nc, Ns)

    alpha, beta, gamma = define_gains()

    Ip_GPe[0,:] = Te
    Ip_GPi[0,:] = Ti
    Ip_ChI[0] = TChI + gamma * dopamine_tonic

    reward = np.nan
    latest_rewarded_time = 0
    gain = 0.01

    for t in range(simulation_length - 1):

        # production of phasic dopamine
        if np.isnan(reward):
            dopamine_phasic = 0
        else:
            # if time range falls between dopamine active period
            if t >= latest_rewarded_time + latency and t <= latest_rewarded_time + latency + duration:
                dopamine_phasic = reward * dopamine_tonic
            else:
                dopamine_phasic = 0
        dopamine_net = dopamine_phasic + dopamine_tonic

        # find sigmoid activations
        act_C[t,:] = sigmoid(Ip_C[t,:])
        act_Go[t,:] = sigmoid(Ip_Go[t,:])
        act_NGo[t,:] = sigmoid(Ip_NGo[t,:])
        act_GPe[t,:] = sigmoid(Ip_GPe[t,:])
        act_GPi[t,:] = sigmoid(Ip_GPi[t,:])
        act_ChI[t] = sigmoid(Ip_ChI[t])
        act_Th[t,:] = sigmoid(Ip_Th[t,:])
        act_STN[t] = sigmoid(Ip_STN[t])

        #winners = np.where( act_C >= winning_threshold)[0]
        winners = np.argmax(act_C)
        if winners.size > 0:
            if winners == correct_winner:
                reward = 1
            else:
                reward = -1
        else:
            reward = 0
        
        noise = np.random.randn()

        # differential equations governing
        
        Lat_Inh[t+1,:] = Lat_Inh[t,:] + (dt/tauL) * (-Lat_Inh[t,:] + np.dot(Wlat, Ip_C[t,:]))        
        Ip_C[t+1,:] = Ip_C[t,:] + (dt/tau) * (-Ip_C[t,:] + np.dot(Wcs, Stimulus) + Lat_Inh[t,:] + np.dot(Wct, Ip_Th[t,:]) + noise)
        
        Ip_Go_DA[t+1,:] = alpha * dopamine_net * Ip_Go[t,:] + np.dot(W_g_ChI, Ip_ChI[t])
        Ip_NGo_DA[t+1,:] = beta * dopamine_net * Ip_NGo[t,:] + np.dot(W_n_ChI, Ip_ChI[t])
        
        Ip_Go[t+1,:] = Ip_Go[t,:] + (dt/tau) * (-Ip_Go[t,:] + np.dot(Wsg, Stimulus) + np.dot(Wcg, Ip_C[t,:]) + Ip_Go_DA[t,:])
        Ip_NGo[t+1,:] = Ip_NGo[t,:] + (dt/tau) * (-Ip_NGo[t,:] + np.dot(Wsn, Stimulus) + np.dot(Wcn, Ip_C[t,:]) + Ip_NGo_DA[t,:])
        
        Ip_GPe[t+1,:] = Ip_GPe[t,:] + (dt/tau) * (-Ip_GPe[t,:] + np.dot(Wne,Ip_NGo[t,:]) + np.dot(W_e_STN, Ip_STN[t]) + act_GPe[t,:])
        
        Ip_GPi[t+1,:] = Ip_GPi[t,:] + (dt/tau) * (-Ip_GPi[t,:] + np.dot(Wgi, Ip_Go[t,:]) + np.dot(Wei, act_GPe[t,:]) +
                                                                    np.dot(W_STN_i, Ip_STN[t]) + act_GPi[t,:])

        Ip_ChI[t+1] = Ip_ChI[t] + (dt/tau) * (-Ip_ChI[t] + act_ChI[t] + gamma * dopamine_net)
        Ip_Th[t+1,:] = Ip_Th[t,:] + (dt/tau) * (-Ip_Th[t,:] + np.dot(Wit, act_GPi[t,:]) + np.dot(Wtc, act_C[t,:]))
        Ip_STN[t+1] = Ip_STN[t] + (dt/tau) * (-Ip_STN[t] + np.dot(W_eng_STN, Eng[t]) + np.dot(W_e_STN, np.sum(act_GPe[t,:])) +
                                                            np.dot(W_STN_i, np.sum(act_STN[t])))

    delta_Wcg = np.zeros(shape=(Nc,Nc))
    delta_Wcn = np.zeros(shape=(Nc,Nc))
    delta_Wsg = np.zeros(shape=(Nc,Nc))
    delta_Wsn = np.zeros(shape=(Nc,Nc))

    if (latest_rewarded_time + latency + duration) < simulation_length:
        out_GO = np.amin(act_Go[latest_rewarded_time+latency:latest_rewarded_time+latency+duration], axis = 0)
        if reward == 1:
            out_GO[winners] = np.max(act_Go[latest_rewarded_time+latency:latest_rewarded_time+latency+duration, winners])
            out_NoGO = np.amin(act_NGo[latest_rewarded_time+latency:latest_rewarded_time+latency+duration], axis = 0)
        else:
            out_NoGO = np.amax(act_NGo[latest_rewarded_time+latency:latest_rewarded_time+latency+duration], axis = 0)
    threshold_hebb = 0.5
    # update delta of learnable synapses
    for r in range(Nc):
        for c in range(Nc):
            if r == c:
                delta_Wcg[r,c] = gain * max(0,( (act_C[latest_rewarded_time,r] - threshold_hebb) * (out_GO[r] - threshold_hebb)))
                delta_Wcn[r,c] = gain * max(0,( (act_C[latest_rewarded_time,r] - threshold_hebb) * (out_NoGO[r] - threshold_hebb)))
            delta_Wsg[r,c] = gain * (out_GO[r] - threshold_hebb) * max(0, Stimulus[c] - threshold_hebb)
            delta_Wsn[r,c] = gain * (out_NoGO[r] - threshold_hebb) * max(0, Stimulus[c] - threshold_hebb)
    
    #post_synaptic weights
    Wcg_post = Wcg + delta_Wcg
    Wcn_post = Wcn + delta_Wcn
    Wsg_post = Wsg + delta_Wsg
    Wsn_post = Wsn + delta_Wsn

    return reward, Wcg_post, Wcn_post, Wsg_post, Wsn_post

if __name__  == "__main__":
    N_stimuli = 4
    # stimulus 1
    Stimulus_1 = np.zeros(N_stimuli)
    Stimulus_1[0] = 1

    # stimulus 2
    Stimulus_2 = np.zeros(N_stimuli)
    Stimulus_2[1] = 1

    # stimulus 3
    Stimulus_3 = np.zeros(N_stimuli)
    Stimulus_3[1] = 1

    # stimulus 4
    Stimulus_4 = np.zeros(N_stimuli)
    Stimulus_4[1] = 1

    N_channels = 4

    Synapse_cortex_GO = 0.5 * np.ones(shape =(N_channels,N_channels))
    Synapse_cortex_NOGO = 0.5 * np.ones(shape =(N_channels,N_channels))
    Synapse_stimuli_GO = 0.5 * np.ones(shape =(N_channels,N_channels))
    Synapse_stimuli_NOGO = 0.5 * np.ones(shape =(N_channels,N_channels))

    N_epochs = 10
    Dopamine_tonic = 1.2

    Synapse_cortex_GO_monitor = np.zeros(shape = (N_epochs, N_channels, N_channels))
    Synapse_cortex_NOGO_monitor = np.zeros(shape = (N_epochs, N_channels, N_channels))
    Synapse_stimuli_GO_monitor = np.zeros(shape = (N_epochs, N_channels, N_channels))
    Synapse_stimuli_NOGO_monitor = np.zeros(shape = (N_epochs, N_channels, N_channels))

    Synapse_cortex_GO_monitor
    Epoch_reward = np.zeros(N_epochs)
    Epoch_punishment = np.zeros(N_epochs)
    Epoch_noresponse = np.zeros(N_epochs)

    Epoch_S = np.zeros(shape=(N_epochs, N_stimuli))
    Epoch_S1 = np.zeros(shape=(N_epochs, N_stimuli))
    Epoch_S2 = np.zeros(shape=(N_epochs, N_stimuli))
    Epoch_S3 = np.zeros(shape=(N_epochs, N_stimuli))
    Epoch_S4 = np.zeros(shape=(N_epochs, N_stimuli))

    Current_stimulus = Stimulus_1
    Current_winner = 1
    for i in range(0, N_epochs - 1):
        
        Synapse_cortex_GO = Synapse_cortex_GO_monitor[i]
        Synapse_cortex_NOGO = Synapse_cortex_NOGO_monitor[i]
        Synapse_stimuli_GO = Synapse_stimuli_GO_monitor[i]
        Synapse_stimuli_NOGO = Synapse_stimuli_NOGO_monitor[i]

        Stimulus_choose = i % 4
        
        noise = np.random.randn()

        if Stimulus_choose == 0:
            Current_stimulus = Stimulus_1
            Current_winner = 1
        
        if Stimulus_choose == 1:
            Current_stimulus = Stimulus_2
            Current_winner = 2
        
        if Stimulus_choose == 2:
            Current_stimulus = Stimulus_3
            Current_winner = 3
        
        if Stimulus_choose == 3:
            Current_stimulus = Stimulus_4
            Current_winner = 4

        Current_stimulus += noise

        Current_stimulus[Current_stimulus > 1] = 1
        Current_stimulus[Current_stimulus < 0] = 0
        
        reward, Synapse_cortex_GO_post, Synapse_cortex_NOGO_post, Synapse_stimuli_GO_post, Synapse_stimuli_NOGO_post = BG(Current_stimulus, N_channels, Synapse_stimuli_GO, Synapse_stimuli_NOGO, Synapse_cortex_GO, Synapse_cortex_NOGO, Dopamine_tonic, Current_winner)

        if reward == 1:
            Epoch_reward[i] = 1
        elif reward == -1:
            Epoch_punishment[i] = 1
        else:
            Epoch_noresponse[i] = 1
        
        Synapse_stimuli_GO_monitor[i+1] = Synapse_stimuli_GO_post
        Synapse_stimuli_NOGO_monitor[i+1] = Synapse_stimuli_NOGO_post
        Synapse_cortex_GO_monitor[i+1] = Synapse_cortex_GO_post
        Synapse_cortex_NOGO_monitor[i+1] = Synapse_stimuli_NOGO_post