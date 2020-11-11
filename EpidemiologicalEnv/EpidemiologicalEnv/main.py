import gym
import json

#from stable_baselines import TRPO
from EpidemiologicalEnv.envs.EpidemiologicalEnv import EpidemiologicalEnv
import pandas as pd

import gym

import matplotlib.pyplot as plt

import math
import torchvision.transforms as T
import numpy as np

import time
from DQN import DQN

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

def plot_res(values, title=''):   
    ''' Plot the reward curve and histogram of results over time.'''
    # Update the window after each episode
    #clear_output(wait=True)
    
    averaged_values = []
    k=100
    for i in range(len(values)):
        if i<k:
            averaged_values.append(values[i])
        else:
            averaged_values.append(sum(values[i-k:i])/k)
    #print(values)
    #print(averaged_values)
        
    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
    f.suptitle(title)
    ax.plot(averaged_values, label='score per run')
    ax.axhline(195, c='red',ls='--', label='goal')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')
    x = range(len(averaged_values))
    ax.legend()
    f.tight_layout(pad=3.0)
    # Calculate the trend
    try:
        z = np.polyfit(x, averaged_values, 1)
        p = np.poly1d(z)
        ax[0].plot(x,p(x),"--", label='trend')
    except:
        print('')
    
   
    plt.show()




def random_search(env, episodes, 
                  title='Random Strategy'):
    """ Random search strategy implementation."""
    final = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        total = 0
        while not done:
            # Sample random actions
            action = env.action_space.sample()
            # Take action and extract results
            next_state, reward, done, _ = env.step(action)
            # Update reward
            total += reward
            if done:
                break
        # Add to the final reward
        final.append(total)
        plot_res(final,title)
    return final


def q_learning(env, model, episodes=1000, gamma=0.95, 
               epsilon=0.02, eps_decay=0.99,
               replay=False, replay_size=10000, 
               title = 'DQL', double=False, 
               n_update=10, soft=False, verbose=True):
    """Deep Q Learning algorithm using the DQN. """
    final = []
    memory = []
    episode_i=0
    last = False
    i=0
    sum_total_replay_time=0
    for episode in range(episodes):
        if i==episodes-1:
            last=True
        episode_i+=1
        if double and not soft:
            # Update target network every n_update steps
            if episode % n_update == 0:
                model.target_update()
        if double and soft:
            model.target_update()
        
        # Reset state
        state = env.reset()
        done = False
        total = 0
        
        while not done:
            # Implement greedy search policy to explore the state space
            if random.random() >= epsilon or last:
                q_values = model.predict(state)
                #print("Q values==",q_values)
                #rint("torch.argmax(q_values)",torch.argmax(q_values))

                action = torch.argmax(q_values).item()
                #print("Predicted action=",action)
            else:
                action = env.action_space.sample()
                #print("Sampled action",action)
            # Take action and add reward to total
            next_state, reward, done, _ = env.step(action)
            
            # Update total and memory
            total += reward
            memory.append((state, action, next_state, reward, done))
            if len(memory)>10000:
                memory = memory[len(memory)-10000:len(memory)]
            q_values = model.predict(state).tolist()
             
            if done:
                if not replay:
                    q_values[action] = reward
                    # Update network weights
                    #print("state==",state)
                    #print("q value==",q_values)
                    model.update(state, q_values)
                break

            if replay:
                t0=time.time()
                # Update network weights using replay memory
                model.replay(memory, replay_size, gamma)
                t1=time.time()
                sum_total_replay_time+=(t1-t0)
            else: 
                # Update network weights using the last step only
                q_values_next = model.predict(next_state)
                #print(q_values)
                #print(action)
                q_values[action] = reward + gamma * torch.max(q_values_next).item()
                #print("state==",state)
                #print("q value==",q_values)
                model.update(state, q_values)

            state = next_state
        
        # Update epsilon
        epsilon = max(epsilon * eps_decay, 0.02)
        final.append(total)
        if episode ==episodes-1:
            print("plotting reward")
            plot_res(final, title)
        
        if verbose:
            print("episode: {}, total reward: {}".format(episode_i, total))
            if replay:
                print("Average replay time:", sum_total_replay_time/episode_i)
        i+=1
    return final


#From tut
env = gym.make('EpidemiologicalEnv:epidemiological-env-v0')
# Number of states
n_state = env.observation_space.shape[0]
# Number of actions
n_action = env.action_space.n
# Number of episodes
episodes =3
# Number of hidden nodes in the DQN
n_hidden = 64
# Learning rate
lr = 0.001
'''
#for stable baseline env
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN


model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=20000)

obs = env.reset()
i=0
while i<200:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    i+=1
env.render()
'''

#uncomment for stable baselines PPO
'''
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
model = PPO2(MlpPolicy, env, verbose=0,n_steps= 64, gamma=0.9, learning_rate= 0.0022763966820360335, ent_coef=0.0005184184762068039, cliprange=0.3, noptepochs= 50,lam=0.98)
model.learn(total_timesteps=200000)
obs = env.reset()
for i in range(200):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
env.render()
'''

#Custom DQN
simple_dqn = DQN(n_state, n_action, n_hidden, lr)
simple = q_learning(env, simple_dqn, episodes, gamma=.95, epsilon=0.05,replay=False)
env.render()
