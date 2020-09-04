import gym
import json
import datetime as dt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.StockTradingEnv import StockTradingEnv

import pandas as pd

df = pd.read_csv('./data/AAPL.csv')
df = df.sort_values('Date')

# The algorithms require a vectorized environment to run
print("Main 1...........")
env = DummyVecEnv([lambda: StockTradingEnv(df)])
print("Main 2...........")
model = PPO2(MlpPolicy, env, verbose=1)
print("Main 3...........")
model.learn(total_timesteps=20)
print("Main 4...........")
obs = env.reset()
print("Main 5...........")
for i in range(20):
    print("Main 6...........")
    action, _states = model.predict(obs)
    print("Main 7...........")
    obs, rewards, done, info = env.step(action)
    print("Main 8...........")
    env.render()

