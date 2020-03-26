import torch
import numpy as np

class JK(torch.nn.Module):
    """
        JK flip-flop Neural Network to solve sequential problems
        ....

        Attributes
        ----------
        
        input_dim, output_dim : (int) Input and output neurons
        nh1, nh2:               (int) First and second layer neurons
        nh3:                    (int) Number of flipflops (for brevity nh2 = 2 x nh3)
        j_w, k_w:               (tensor double) weights of j and k
        l1, l2, l4:             (torch.nn.Module) linear layers
        out1, out2, out4:       (torch tensor) output of the l1, l2, and l4 respectively
        j, k:                   (torch tensor) j and k (input) for flipflop
        inp_j, inp_k:           (torch tensor) weighted input j and k
        out3:                   (torch tensor) output of flipflop neurons
        vstate:                 (torch double) stores state values of all flipflops
        .....
        
        Notes
        -----
            1. The output V(t) is calculated as V(t) = (1 - V(t-1))* J + V(t-1)*(1 - K)
            2. shape of vstate is [no_of_timesteps, no_of_samples, no_of_flipflops]
                For eg: vstate[0,1,2] denotes state value of 0th timestep (state) of 1st data sample at 2nd flipflop
            3. To modify the architecture, necessary changes shall be done easily by extending/shortening
                the variables used. 
                For eg: To add one more flipflop layer, variables like nh4, l4, j, k etc. should be added and
                importantly, the changes should be reflected in forward() and backward()
        .....

        Usage
        -----
        model = JK(input_dim, nh1, nh2, nh3, output_dim)
        output = model(input)
        loss = LossFunction(output, target)
        optimizer.zero_grad()
        model.backward(loss, output, target, input)
        optimizer.step()
        .....

        References
        ----------
        1) Decision making with long delays using networks of flip-flop neurons. 
           Pavan Holla ; Srinivasa Chakravarthy 
        2) https://www.electronics-tutorials.ws/sequential/seq_2.html
        3) https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c 
        .....

        TODO: Check vanishing/exploding of gradients
        TODO: Check concat/dot product methods as in LSTM
    """

    def __init__(self, input_dim, nh1, nh2, nh3, output_dim):
        """
            Constructor to initialize all necessary entities
        """
        # inherit from super class: torch.nn.Module
        super(JK, self).__init__()
        self.input_dim = input_dim
        self.nh1 = nh1
        self.nh2 = nh2
        self.nh3 = nh3
        self.output_dim = output_dim
        # check two input to one flipflop
        if self.nh2 // 2 != self.nh3:
            raise Exception("Invalid dimensional input for flipflop layer")
        self.j_w = torch.randn(self.nh2 // 2, requires_grad = True).double()
        self.k_w = torch.randn(self.nh2 // 2, requires_grad = True).double()
        self.l1 = torch.nn.Linear(self.input_dim, self.nh1)
        self.l2 = torch.nn.Linear(self.nh1, self.nh2)
        # l3 is the flipflop layer, so avoiding explicit namespace here
        self.l4 = torch.nn.Linear(self.nh3, self.output_dim)

    def sigmoid(self, X, norm = True, a = 20):
        """
            Sigmoid function, when `norm = True` normalization (min/max) is done
        """
        if norm == True:
            X = (X - torch.min(X)) / (torch.max(X) - torch.min(X))
        return 1 / (1 + torch.exp(-1 * a * X))

    def sigmoid_derivative(self, X, a = 20):
        """
            First derivative of sigmoid
        """
        return a * self.sigmoid(X, True) * (1 - self.sigmoid(X, True))

    def forward(self, X):
        """
            Forward propogation of data across the layers
        """
        # store the output of the model for each time instant
        output_signal = []
        self.vstate = torch.zeros(size = (X.size(1), X.size(0), self.nh3))   # nh3 is equal to number of flipflops
        if self.vstate.shape[2] != self.nh3:
            raise Exception("Invalid dimensional input for flipflop layer")    
        # chunk the input data to feed the model with data of particular instant (of all samples)
        for i, x in enumerate(X.chunk(X.size(1), dim = self.input_dim)):
            self.out1 = self.sigmoid(self.l1(x))
            self.out2 = self.sigmoid(self.l2(self.out1))
            self.j = self.out2[: , 0 : self.out2.shape[1] // 2]
            self.k = self.out2[: , self.out2.shape[1] // 2 : self.out2.shape[1]]
            self.inp_j = self.sigmoid(self.j * self.j_w)
            self.inp_k = self.sigmoid(self.k * self.k_w)
            # for t = 0; t-1 doesn't exists
            if i == 0:
                current_state = self.vstate[i]
            else:
                current_state = self.vstate[i - 1]
            current_state = current_state.double()
            self.out3 = torch.tensor(((1 - current_state) * self.inp_j) + ((1 - self.inp_k) * current_state)).double()
            # store it in vstate to use it in next time step
            self.vstate[i] = self.out3
            self.out4 = self.sigmoid(self.l4(self.out3))
            # append the output data for each time instant
            output_signal += [self.out4]
        # reshape to make consistent with input data shape/ target shape
        output_signal = torch.stack(output_signal, 1).squeeze(2)
        return output_signal

    def backward(self, loss, y, targ, x):
        """
            Backpropogation of error across each layer to estimate weight_grad using chain-rule 
            of differentiation [Ref-3]
            a_b defines da/db i.e. gradient of loss wrt kth layer is denoted as loss_l(k)w
        """
        # loss_l4w = loss_out4 * out4_out3 * out3_l4w
        loss_out4 = (-2 / y.shape[0])  * torch.sum(targ - y)
        out4_out3 = (1 / self.out3.shape[0]) * torch.sum(self.sigmoid_derivative(self.out3), axis = 0)
        out3_l4w = (1 / self.out3.shape[0]) * torch.sum(self.out3, axis = 0)
        loss_l4w = loss_out4 * out4_out3 * out3_l4w

        # loss_(j/k)w = loss_out3 * out3_inp(j/k) * inp(j/k)_(j/k)
        loss_out3 = (loss_out4 * out4_out3).double()
        out3_inpj = (1 / self.vstate.shape[0]) * torch.sum(1 - self.vstate[self.vstate.shape[0] - 1], axis = 0).double()
        out3_inpk = (1 / self.vstate.shape[0]) * torch.sum(-1 * self.vstate[self.vstate.shape[0] - 1], axis = 0).double()
        inpj_j = ( 1 / self.j.shape[0]) * torch.sum(self.j, axis = 0).double()
        inpk_k = ( 1 / self.k.shape[0]) * torch.sum(self.k, axis = 0).double()
        loss_jw = loss_out3 * (out3_inpj.double() * inpj_j.double()).double()
        loss_kw = loss_out3 * (out3_inpk.double() * inpk_k.double()).double()
        
        # loss_l2w = loss_out2 * out2_out1 * out1_l2w, 
        # where loss_out2 = concat((loss_out3 * out3_inpj * inpj_j), (loss_out3 * out3_inpk * inpk_k)) 
        loss_out2_j = torch.matmul( torch.matmul(loss_out3.reshape(1, -1),out3_inpj.reshape(-1, 1)), inpj_j.reshape(1, -1)).reshape(-1)
        loss_out2_k = torch.matmul( torch.matmul(loss_out3.reshape(1, -1), out3_inpk.reshape(-1, 1)), inpk_k.reshape(1, -1)).reshape(-1)
        loss_out2 = torch.cat((loss_out2_j, loss_out2_k)).double()
        out2_out1 = (1 / self.out1.shape[0]) * torch.sum(self.sigmoid_derivative(self.out1), axis = 0)
        out1_l2w = (1 / self.out1.shape[0]) * torch.sum(self.out1, axis = 0)
        loss_l2w = loss_out2.reshape(-1, 1) * (out2_out1 * out1_l2w).reshape(1,-1)
        
        #loss_l1w = loss_out1 * out1_x * x_l1w
        loss_out1 = loss_out2.reshape(-1, 1) * out2_out1.reshape(-1, 1)
        out1_x = (1 / x.shape[0]) * torch.sum(self.sigmoid_derivative(x), axis = 0)
        x_l1w = (1 / x.shape[0]) * torch.sum(x, axis = 0)        
        loss_l1w = torch.matmul(loss_out1.reshape(-1, 1), torch.matmul(out1_x.reshape(1, -1), x_l1w.reshape(-1, 1))).double()

        # add it to grad tensor to make it utilize for optimizer
        self.l4.weight.grad = loss_l4w.reshape(1, -1)
        self.j_w.grad = loss_jw
        self.k_w.grad = loss_kw
        self.l2.weight.grad = loss_l2w
        self.l1.weight.grad = loss_l1w