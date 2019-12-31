from brian2 import *
tau = 10*ms
w = 10*mV
eqn = """ dv/dt = (1-v)/tau :1"""
stn = NeuronGroup(2, eqn, method = exact)
gpe = NeuronGroup(2, eqn, method = exact)
mon = StateMonitor(stn, 'v')

run(100*ms)
print(mon.v[0])