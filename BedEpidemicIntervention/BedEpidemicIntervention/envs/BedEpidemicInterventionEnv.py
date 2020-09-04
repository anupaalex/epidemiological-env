import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from scipy.integrate import simps
from numpy import trapz
import scipy.integrate as spi
import numpy as np
import pylab as pl
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import re

class BedEpidemicInterventionEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BedEpidemicInterventionEnv, self).__init__()
        self.beta = 0.4482
        self.delta=1
        self.alpha = 0.1923
        self.gamma = 0.1724
        self.theta = 1
        self.lower_theta = 0.5


        self.i_i = 0
        self.s_i = 12600000/(12600000+1000)
        self.e_i = 1000/(12600000+1000)
        self.r_i =0
        self.population = 12600000+1000
        self.infected_list = []
        self.susceptible_list = []
        self.exposed_list = []
        self.recovered_list = []
        self.infected_count_list = []


        self.infected_list.append(self.i_i)
        self.susceptible_list.append(self.s_i)
        self.exposed_list.append(self.e_i)
        self.recovered_list.append(self.r_i) 
        self.infected_count_list.append(0)

        self.days = 200
        self.current_day=0
        self.economy = 10
        self.economy_percent=1
        self.bed_per_1000 = 1.5
        self.lower_bed_per_1000 = 0.5
        self.num_beds = (self.bed_per_1000/1000) * 12600000
        self.reward =0
        self.rewards = []
        self.action_list = []
        self.action_string = ""

        self.reduce_bed = False
        self.reduce_theta = False
        self.limit_changes = True

        self.bed_decrease_per_day = (((self.bed_per_1000/1000)-(self.lower_bed_per_1000/1000)) * 12600000)/self.days
        self.theta_decrease_per_day = (self.theta - self.lower_theta)/self.days
        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=100000000, shape=(25,), dtype=np.float32)

    def diff_eqs(self,INP,t):  
        '''The main set of equations'''
        Y=np.zeros((3))
        V = INP    
        Y[0] = - self.beta*self.delta*self.theta * V[0] * V[2] 
        Y[1] = self.beta*self.delta*self.theta * V[0] * V[2] - self.alpha * V[1] 
        Y[2] = self.alpha * V[1] - self.gamma * V[2] 
        return Y   # For odeint

    def _next_observation(self):
        '''
        old_s_i = self.s_i
        old_i_i = self.i_i
        old_e_i = self.e_i
        old_r_i = self.r_i
        
        self.s_i = old_s_i - self.delta*self.beta*old_s_i*old_i_i
        self.e_i = old_e_i + self.delta*self.beta*old_s_i*old_i_i - self.alpha*old_e_i
        self.i_i = old_i_i + self.alpha *old_e_i - self.gamma*old_i_i
        self.r_i = old_r_i + self.gamma * old_i_i
        '''
        INPUT =(self.s_i,self.e_i,self.i_i)
        t_start = 0.0; t_end = 1; t_inc = 1
        t_range = np.arange(t_start, t_end+t_inc, t_inc)

        RES = spi.odeint(self.diff_eqs,INPUT,t_range)
        self.s_i,self.e_i,self.i_i= RES[1][0],RES[1][1],RES[1][2]
        self.r_i = 1 - self.s_i - self.e_i - self.i_i
        self.infected_list.append(self.i_i)
        self.susceptible_list.append(self.s_i)
        self.exposed_list.append(self.e_i)
        self.recovered_list.append(self.r_i)
        if self.current_day<=24:
            obs = [0 for i in range(25-self.current_day)]
            obs.extend(self.infected_list[:self.current_day])
            
        else:
            n = len(self.infected_list)
            obs = self.infected_list[self.current_day-25:self.current_day]
        return obs

    def reduce_bed_func(self):
        if self.reduce_bed:
            self.num_beds = self.num_beds-self.bed_decrease_per_day

    def reduce_theta_func(self):
        if self.reduce_theta:
            self.theta = self.theta-self.theta_decrease_per_day

    def limit_changes_func(self):
        if self.limit_changes:
            pattern = re.compile(r"0*1*2*3*0*$")
            if bool(re.match(pattern,self.action_string)):
                return 0
            else :
                return -10
        return 0
    
    def step(self, action):
        self.action_list.append(action)
        if action == 0:
            self.delta = 0.25
            self.economy_percent = 0.4
        if action == 1:
            self.delta = 0.5
            self.economy_percent = 0.6
        if action == 2:
            self.delta = 0.75
            self.economy_percent = 0.8
        if action == 3:
            self.delta = 1
            self.economy_percent = 1

        beds_required = 0.05*self.i_i*(12600000+1000)
        self.infected_count_list.append(beds_required)
        #print("step : %d action: %d num beds: %f num infected in icu: %f "%(self.current_day, action,self.num_beds,infected))
        return_reward= 0

        self.reduce_theta_func()
        self.reduce_bed_func()

        self.action_string = self.action_string + str(action)

        return_reward = self.limit_changes_func()

        if beds_required>self.num_beds:

            self.reward -= 1000
            return_reward-= 1000
        else :
            self.reward += self.economy_percent*self.economy
            return_reward+= self.economy_percent*self.economy

        self.rewards.append(self.reward)
        self.current_day +=1

        obs = self._next_observation()
        if self.current_day == self.days:
            done = True
            
        else:
            done= False

        return obs, return_reward,done,{}

    def reset(self):
        self.s_i = 12600000/(12600000+1000)
        self.e_i = 1000/(12600000+1000)
        self.num_beds = (1.5/1000) * 12600000
        self.i_i = 0
        self.r_i =0
        self.infected_list = []
        self.infected_count_list = []
        self.susceptible_list = []
        self.exposed_list = []
        self.recovered_list = []
        self.current_day=0
        self.reward =0
        self.rewards = []
        self.action_list = []
        self.economy_percent=1
        self.theta = 1
        obs = self._next_observation()
        return obs

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print("Reward = ",self.reward)
        pl.subplot(211)
        pl.plot(self.infected_count_list, '-g', label='ICU with intervention')
    
        if not self.reduce_bed:    
            pl.axhline(y=self.num_beds)
        else:
            axes = pl.gca()
            x_vals = np.array([i for i in range(self.days+1)])
            slope =self.bed_decrease_per_day
            intercept = (self.bed_per_1000/1000) * 12600000
            y_vals = intercept - slope * x_vals
            plt.plot(x_vals, y_vals, '--')
        pl.legend(loc=0)
        pl.xlabel('Time')
        pl.ylabel('Beds Required')

        pl.subplot(212)
        pl.plot(self.action_list, '-r', label='Action')
        pl.xlabel('Time')
        pl.ylabel('Action')
        pl.show()
        return self.action_list