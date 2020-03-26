from JKFF import JK

import numpy as np
import torch
from matplotlib import pyplot as plt

def generate_sinewave():
    time_period = 5
    length = 100
    n = 100
    X = np.empty((n, length))
    X[:] = np.array(range(length)) + np.random.randint(-10 * time_period, 10 * time_period, n).reshape(n, 1)
    data = np.sin(X / 1.0 / time_period).astype('float64')
    return data

data = generate_sinewave()
inp = torch.from_numpy(data[:, :-1])
target = torch.from_numpy(data[:, 1:])

model = JK(1, 100, 100, 50, 1)
model.double()
inp = model.sigmoid(inp)
target = model.sigmoid(target)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adamax(model.parameters() ,lr = 0.0001)

epochs = 20
out_mon = []
pred_mon = []
mse_monitor = []

for i in range(epochs):

    out = model(inp)    
    loss = criterion(out, target)
    model.backward(loss, out, target, inp)
    optimizer.step()
    optimizer.zero_grad
    print("Loss = {}".format(loss.item()))
    o = out[99].clone().detach().numpy()
    p = target[99].clone().detach().numpy()
    out_mon.append(o)
    pred_mon.append(p)
    mse_monitor.append(loss.item())

    plt.plot(out_mon[i], label = "Predicted")  
    plt.plot(pred_mon[i], label = "Original")  
    plt.legend()
    plt.show()