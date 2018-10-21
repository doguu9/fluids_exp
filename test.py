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
fluids.OBS_GRID
fluids.OBS_BIRDSEYE
fluids.OBS_GRID
fluids.OBS_NONE

if __name__ == '__main__':

    sup = FluidsSupervisor()
    env = FluidsEnv(fluids.OBS_GRID)


    states, int_actions, _, reward, infos = statistics.collect_traj(env, sup, 100, False)
    IPython.embed()



    # lnr_rewards = []
    # for i in range(iterations):
    #     env = gym.make("fluids-v2")
    #     sup = FluidsSupervisor(gym_fluids.agents.fluids_supervisor)
    #     states, tmp_actions, _, reward = statistics.collect_traj(env, sup, 100, True)

    #     # train model
    #     # 

    #     states, tmp_actions, _, lnr_reward = statistics.collect_traj(env, lnr, 100, True)
    #     lnr_rewards.append(lnr_reward)

    # IPython.embed()
