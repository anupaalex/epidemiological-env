import gym
import json
import datetime as dt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from envs.EpidemicInterventionEnv import EpidemicInterventionEnv

import pandas as pd


import os

import gym
import numpy as np
import matplotlib.pyplot as plt

#from stable_baselines import DDPG
#from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback
import shutil

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

# Create log dir
log_dir = "tmp2/"
shutil.rmtree(log_dir)
os.makedirs(log_dir, exist_ok=True)



df = pd.read_csv('./data/epidemic_2.csv')

df['Date_new'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df.sort_values('Date_new')
env = gym.make('EpidemicIntervention:epidemic-intervention-v0')

#new
env = Monitor(env, log_dir)

# Add some param noise for exploration
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)
# Because we use parameter noise, we should use a MlpPolicy with layer normalization


'''
env = DummyVecEnv([lambda: EpidemicInterventionEnv(df)])
'''

model = PPO2(MlpPolicy, env, verbose=1)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000000, log_dir=log_dir)
# Train the agent
time_steps = 1e5

model.learn(total_timesteps=1000000,callback=callback)
results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "DDPG LunarLander")
plt.show()

obs = env.reset()
action_values =set()
env.current_step =0
print("!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(env.current_step)


for i in range(250):
    print("i============",i)
    action, _states = model.predict(obs)
    print(action)
    env.reset()
    obs, rewards, done, info = env.step(action)
    if done == True:
    	break
    	env.reset
    #ret_df = env.render()
    #action_values.add(action[0][0])
ret_df = env.render()
print(action_values)

import pylab as pl 
#pl.plot(list(ret_df['Infected']), '-g', label='Susceptibles')
#pl.show()
old_df = pd.read_csv('./data/epidemic_2.csv')

pl.subplot(211)
pl.plot(ret_df['Susceptible'], '-g', label='Susceptibles')
pl.plot(ret_df['Recovered'], '-k', label='Recovereds')
pl.plot(df['Susceptible'], '-b', label='Susceptibles Old')
pl.plot(df['Recovered'], '-b', label='Recovereds Old')
pl.legend(loc=0)

pl.xlabel('Time')
pl.ylabel('Susceptibles and Recovereds')
pl.subplot(212)
pl.plot(ret_df['Infected'], '-r', label='Infectious')
pl.plot(df['Infected'], '-b', label='Infectious Old')
pl.xlabel('Time')
pl.ylabel('Infectious')
pl.show()