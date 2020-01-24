import torch
import numpy as np
from matplotlib import pyplot as plt

"""
TODO: Add multiple layers of SR ffs
TODO: Add effecient and resonable backprop mechansim across ff layer
TODO: Replace SR with universal ffs
TODO: Add documentation and refs across
"""

def generate_sinewave():
    time_period = 20
    length = 1000
    n = 10
    X = np.empty((n, length))
    X[:] = np.array(range(length)) + np.random.randint(-10 * time_period, 10 * time_period, n).reshape(n, 1)
    data = np.sin(X / 1.0 / time_period).astype('float64')
    return data

class FFlayer(torch.nn.Module):

    def __init__(self, input_dim, output_dim):    
        super(FFlayer, self).__init__()
        self.input_dim = input_dim
        if self.input_dim != output_dim * 2:
            raise Exception("Invalid dimensional input for flipflop layer")
        self.output_dim = output_dim
          
    def forward(self, input, vstate):
        if vstate.shape[1] != self.output_dim:
            raise Exception("Invalid dimensional input for flipflop layer")    
        s = input[: , [col % 2 == 0 for col in range(input.shape[1])]].clone().detach().numpy()
        r = input[: , [col % 2 != 0 for col in range(input.shape[1])]].clone().detach().numpy()
        vstate = vstate.clone().detach().numpy()
        return torch.tensor(vstate - 2 * s + 0.5 * r + 0.5)

class Model(torch.nn.Module):

    def __init__(self, input_dim, nh1, nh2, nh3, nh4, output_dim):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.nh1 = nh1
        self.nh2 = nh2
        self.nh3 = nh3      
        self.nh4 = nh4
        self.output_dim = output_dim
        self.l1 = torch.nn.Linear(self.input_dim, self.nh1)
        self.l2 = torch.nn.Linear(self.nh1, self.nh2)
        self.l3 = FFlayer(self.nh2, self.nh3)
        self.l4 = torch.nn.Linear(self.nh3, self.nh4)
        self.l5 = torch.nn.Linear(self.nh4, self.output_dim)

    def normalize(self, input):
        return (input - torch.min(input)) / (torch.max(input) - torch.min(input))

    def forward(self, input):
        output_signal = []
        a = 5 # slope to sigmoid function
        vstate = torch.zeros(size = (input.size(1), input.size(0), self.nh3))   # nh3 is equal to number of flipflops
        for i, input_t in enumerate(input.chunk(input.size(1), dim = self.input_dim)):

            out1 = torch.sigmoid(a * self.l1(input_t))
            out1 = self.normalize(out1)
            
            out2 = torch.sigmoid(a * self.l2(out1))
            out2 = self.normalize(out2)

            if i > 0:
                out3 = self.l3(out2, vstate[i - 1])
            else:
                out3 = self.l3(out2, vstate[i])

            vstate[i] = out3
            out3 = self.normalize(out3)

            out4 = torch.sigmoid(a * self.l4(out3.double()))
            out4 = self.normalize(out4)
            
            out5 = torch.sigmoid(a * self.l5(out4))
            output_signal += [out5]

        output_signal = torch.stack(output_signal, 1).squeeze(2)
        return output_signal

data = generate_sinewave()
inp = torch.from_numpy(data[:, :-1])
target = torch.from_numpy(data[:, 1:])
model = Model(1, 50, 100, 50, 20, 1)
model.double()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
epochs = 10
# performance monitors
out_mon = []
pred_mon = []
mse_monitor = []
    
for i in range(epochs):
    optimizer.zero_grad()
    out = model(inp)
    loss = criterion(out, target)
    loss.backward()
    print("Loss = {}".format(loss.item()))
    o = out[0].clone().detach().numpy()
    p = target[1].clone().detach().numpy()
    
    out_mon.append(o)
    pred_mon.append(p)
    mse_monitor.append(loss.item())
    
    optimizer.step()
    plt.plot(out_mon[i], label = "Predicted")  
    plt.legend()
    plt.show()
    # optimize the model
