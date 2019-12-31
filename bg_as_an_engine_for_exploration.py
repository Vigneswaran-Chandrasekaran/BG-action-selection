import numpy as np

def sigmoid(act, gain = 1):
    return( 1 / (1 + np.exp(-(gain * act))))

I_ext = np.array([[1,0],[0,1]])
N_s = 4 # number of MSNs in Striatum

W = np.random.randn(I_ext.shape[0], N_s) 

act = np.sum(np.dot(I_ext, W), axis = 0)

gain = 1.4

act_D1 = sigmoid(act[0:2], gain)
act_D2 = sigmoid(act[2:4], 1/gain)

Wse = np.random.randn(2,2)
Wes = -1 * Wse
Wd2e = np.random.randn(2,2)

act_STN = np.zeros(2)
act_GPE = np.zeros(2)

for i in range(10):
    
    act_GPE = sigmoid(np.dot(act_D2, Wd2e) + np.dot(act_STN, Wd2e))

    act_STN = sigmoid(np.dot(act_GPE, Wes))

    print(act_GPE, act_STN)
    

