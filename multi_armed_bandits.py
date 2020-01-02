"""
Solving Multi-armed bandit problem using e-greedy algorithm
"""
import numpy as np
from matplotlib import pyplot as plt

class BernoulliBandit():
    """
    Class generates Bernoulli Probability Distribution and assign 
    reward function
    """
    def __init__(self, n = 100):
        # number of slot machines to use
        self.n = n
        self.prob_dist = [np.random.random() for _ in range(self.n)]
        self.best_rv = max(self.prob_dist)

    def get_reward(self, i):
        if np.random.random() <= self.prob_dist[i]:
            return 1
        return 0
    
    def get_underlying_prob(self, i):
        return self.prob_dist[i]

class EpsilonGreedySolver():
    """
    Using Epsilon-Greedy Algorithm to solve Multi-armed bandit problem
    """
    def __init__(self, bandit, eps = 0.15, init_probability = 0.95):
        
        assert isinstance(bandit, BernoulliBandit)
        self.bandit = bandit
        self.eps = eps
        self.action_counts = [0] * bandit.n
        self.action_selected = []
        self.net_punishment = 0
        self.hist_punishment = []
        self.initial_assumption = 1.0
        self.estimates = [init_probability] * self.bandit.n
        self.net_reward = 0
        self.hist_reward = []

    def compute_regret(self, i):
        return self.bandit.best_rv - self.bandit.get_underlying_prob(i)

    def return_greedy_step(self):
        if np.random.random() < self.eps:
            return np.random.randint(0, self.bandit.n)
        i = self.estimates.index(max(self.estimates))

        self.estimates[i] += 1/(self.action_counts[i] + 1) * (self.bandit.get_reward(i) - self.estimates[i])
        return i

    def start_stimulation(self, num_steps = 100):
        for _ in range(num_steps):
            i = self.return_greedy_step()
            self.action_counts[i] += 1
            self.action_selected.append(i)
            self.net_punishment += self.compute_regret(i)
            self.hist_punishment.append(self.net_punishment)
            self.net_reward += self.bandit.get_reward(i)
            self.hist_reward.append(self.net_reward)

if __name__ == '__main__':

    bandit = BernoulliBandit()
    agent1 = EpsilonGreedySolver(bandit, eps = 0)
    agent2 = EpsilonGreedySolver(bandit, eps = 0.15)
    agent3 = EpsilonGreedySolver(bandit, eps = 0.35)
    agent4 = EpsilonGreedySolver(bandit, eps = 0.5)
    
    agent1.start_stimulation()
    agent2.start_stimulation()
    agent3.start_stimulation()
    agent4.start_stimulation()
    
    plt.plot(agent1.hist_punishment, label = "Agent1: e=0")
    plt.plot(agent2.hist_punishment, label = "Agent2: e=0.15")
    plt.plot(agent3.hist_punishment, label = "Agent3: e=0.35")
    plt.plot(agent4.hist_punishment, label = "Agent4: e=0.5")
    plt.title("Cumulative Punishments vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cumulative Punishments")
    plt.legend()
    plt.show()

    plt.plot(agent1.hist_reward, label = "Agent1: e=0")
    plt.plot(agent2.hist_reward, label = "Agent2: e=0.15")
    plt.plot(agent3.hist_reward, label = "Agent3: e=0.35")
    plt.plot(agent4.hist_reward, label = "Agent4: e=0.5")
    plt.title("Cumulative Rewards vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cumulative Rewards")
    plt.legend()
    plt.show()




