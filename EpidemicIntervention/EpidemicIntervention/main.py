import gym
import json
import datetime as dt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from envs.EpidemicInterventionEnv import EpidemicInterventionEnv

import pandas as pd

df = pd.read_csv('./data/epidemic_2.csv')

df['Date_new'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df.sort_values('Date_new')
env = gym.make('EpidemicIntervention:epidemic-intervention-v0')

'''
env = DummyVecEnv([lambda: EpidemicInterventionEnv(df)])
'''

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000)

obs = env.reset()
action_values =set()
env.current_step =0
print("!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(env.current_step)


for i in range(250):
    print("i============",i)
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, done, info = env.step(action)
    #ret_df = env.render()
    #action_values.add(action[0][0])
ret_df = env.render()
print(action_values)

import pylab as pl 
#pl.plot(list(ret_df['Infected']), '-g', label='Susceptibles')
#pl.show()


pl.subplot(211)
pl.plot(ret_df['Susceptible'], '-g', label='Susceptibles')
pl.plot(ret_df['Recovered'], '-k', label='Recovereds')
pl.legend(loc=0)

pl.xlabel('Time')
pl.ylabel('Susceptibles and Recovereds')
pl.subplot(212)
pl.plot(ret_df['Infected'], '-r', label='Infectious')
pl.xlabel('Time')
pl.ylabel('Infectious')
pl.show()