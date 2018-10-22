import gym
from tools.supervisor import FluidsSupervisor
from tools.learner import FluidsLearner
from tools import statistics
import IPython
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
import os
import pickle
import argparse
import fluids
from fluids_env import FluidsEnv, FluidsVelEnv
import IPython
import gym_fluids
from tools.lrc import LRC
fluids.OBS_GRID
fluids.OBS_BIRDSEYE
fluids.OBS_GRID
fluids.OBS_NONE

path = os.getcwd()
iterations = 10
alpha = 0.1
eta = 1.0
t = .01
regret = True

sup = FluidsSupervisor()
lnr = FluidsLearner(LRC(alpha, eta, intercept=False), sup)
env = FluidsEnv(fluids.OBS_GRID)

data_states = []
data_actions = []
sup_reward_arr = []
reward_arr = []

for iteration in range(iterations):
    states, intended_actions, taken_actions, reward, infos = statistics.collect_traj(env, sup, 10, True)
    sup_reward_arr.append(reward)
    states, intended_actions, taken_actions, reward, infos = statistics.collect_traj(env, lnr, 10, True)
    reward_arr.append(reward)
    i_actions = []
    for i in range(len(states)):
        i_actions += [sup.intended_action(states[i], infos[i])]

    data_states += states
    data_actions += i_actions

    lnr.set_data(data_states, data_actions)
    lnr.train()


plt.subplot(111)
plt.title("Rewards")
plt.plot(reward_arr, label='Learner rewards')
plt.plot(sup_reward_arr, label='Supervisor Rewards')
plt.legend()
plt.ylim(0, 20)
filepath = path + 'reward.png'
plt.savefig(filepath)
plt.close()
plt.cla()
plt.clf()

