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
Nc = 4

act_C = np.zeros(shape =(simulation_length, Nc))
act_Go = np.zeros(shape =(simulation_length, Nc))
act_NGo = np.zeros(shape =(simulation_length, Nc))
act_GPe = np.zeros(shape =(simulation_length, Nc))
act_GPi = np.zeros(shape =(simulation_length, Nc))
act_Th = np.zeros(shape =(simulation_length, Nc))
act_STN = np.zeros(simulation_length)
act_ChI = np.zeros(simulation_length)

Ip_C = np.zeros(shape =(simulation_length, Nc))
Ip_Go = act_Go
Ip_NGo = act_NGo
Ip_GPe = act_GPe
Ip_GPi = act_GPi
Ip_Th = act_Th
Ip_STN = act_STN
Ip_ChI = act_ChI

Lat_Inh = np.zeros(shape =(simulation_length, Nc))

Eng = np.zeros(simulation_length)

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

Learnable synapses:
------------------

Wcg: cortex to GO
Wsg: stimulus to GO
Wcn: cortex to NoGO
Wsn: stimulus to NoGO

Gains:
-----
alpha: DA to GO (exh)
beta: DA to NoGO (inh)
gamma: DA to ChI (inh)
"""
Ns = 4   # number of stimulus
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

alpha = 0.75
beta = -1
gamma = -0.5

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
"""

Te = 1
Ti = 3
TChI = 1

dopamine_tonic = 1.7 ###########

latency = int(100/dt)
duration = int(50/dt)

Ip_GPe[0,:] = Te
Ip_GPi[0,:] = Ti
Ip_ChI[0] = TChI + gamma * dopamine_tonic

reward = np.nan
latest_rewarded_time = 0

for t in range(simulation_length):

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

    # find activation
    act_C[t,:] = sigmoid(Ip_C[t,:])
    act_Go[t,:] = sigmoid(Ip_Go[t,:])