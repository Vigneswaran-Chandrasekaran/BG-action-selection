from matplotlib import pyplot as plt
import numpy as np
import torch

class MLP(torch.nn.Module):
    """
    Class defining the structure of MLP and layer characterstics
    """
    def __init__(self, input_dim, nh, output_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.nh = nh
        self.output_dim = output_dim
        #layer definition
        self.input_layer = torch.nn.Linear(self.input_dim, self.nh)
        self.output_layer = torch.nn.Linear(self.nh, self.output_dim)
    
    def normalize(self, x, a = 5):
        x = (x - torch.min(x))/ (torch.max(x) - torch.min(x))
        return x * a

    def forward(self, x):
        #propogation of each layer
        self.out1 = torch.nn.functional.relu(self.input_layer(x))
        self.out1 = self.normalize(self.out1)
        self.out2 = torch.nn.functional.sigmoid(self.output_layer(self.out1))
        return self.out2
 
def SRff(s, r, u):
    if s > 0.5 and r > 0.5:
        if s > r:
            return 1
        else:
            return 0
    if s > 0.5 and r <= 0.5:
        return 1
    if s <= 0.5 and r > 0.5:
        return 0
    else:
        return u

def forward_SR(X, n = 75):

    if X.shape[1] != 2 * n:
        raise Exception("Number of flipflops must be half the size of input length")
    # update the states of flipflop for each time instant (here 100 steps)

    v_state = np.zeros((X.shape[0], n))

    for t_instant in range(X.shape[0] - 1):    
        for pair in range(X.shape[1] // 2):
            s = X[t_instant, 2 * pair]
            r = X[t_instant, 2 * pair + 1]
           
            v_state[t_instant + 1, pair] = SRff(s, r, v_state[t_instant, pair])
    return torch.tensor(v_state[v_state.shape[0] - 1]).float()

def generate_stimulus(time_step, arg = 'train'):
    if time_step == 0: 
        if arg == 'test':
            return np.array([0, 1])

        return np.array([1, 0])   # must be equal to reward
    return np.random.uniform(0, 0.2, size = 2)

def prepare_input(time_range = 100, arg = 'train'):
    X = []
    for i in range(time_range):
        X.append(generate_stimulus(i, arg = 'train'))
    X = torch.tensor(X).float()
    return X

mlp1 = MLP(2, 150, 150)

mlp2 = MLP(75, 10, 2)
optimizer = torch.optim.Adam(mlp2.parameters(), lr = 0.001)
reward = torch.tensor([1, 0]).float()

loss_mon = []
for epoch in range(10):

    x = mlp1(prepare_input())
    
    st = forward_SR(x)

    y = mlp2(st)

    loss = torch.sum((y - reward) ** 2)
    loss.backward()
    optimizer.step()
    loss_mon.append(loss)
    print(y)
    if epoch == 99:
        print(y)
        plt.plot(loss_mon, label = "loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()
