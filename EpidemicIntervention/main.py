import gym
import json
import datetime as dt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.EpidemicInterventionEnv import EpidemicInterventionEnv

import pandas as pd

df = pd.read_csv('./data/epidemic_2.csv')

df['Date_new'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df.sort_values('Date_new')
# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: EpidemicInterventionEnv(df)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=20000)

obs = env.reset()
action_values =set()
for i in range(200):
    print("i============",i)
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, done, info = env.step(action)
    ret_df = env.render()
    action_values.add(action[0][0])

print(action_values)
print(ret_df.head())
import pylab as pl
pl.plot(list(ret_df['Infected']), '-g', label='Susceptibles')
pl.show()
