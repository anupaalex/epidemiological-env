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
from scipy.integrate import simps

class EpidemiologicalEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(EpidemiologicalEnv, self).__init__()
        self.beta = 0.4482
        self.delta=1
        self.alpha = 0.1923
        self.gamma = 0.1724
        self.theta = 1
        self.lower_theta = 0.5
        self.deltas=[0.25,0.5,0.75,1]
        self.economy_percents = [0.4,0.6,0.8,1]
        self.cost_percents =[1,0.8,0.6,0.4]
        self.cost = 10
        self.total_cost=0
        self.i_i = 0
        self.s_i = 12600000/(12600000+1000)
        self.e_i = 1000/(12600000+1000)
        self.r_i =0
        self.i_0 = 0
        self.s_0 = 12600000/(12600000+1000)
        self.e_0 = 1000/(12600000+1000)
        self.r_0 =0


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
        self.use_delta = True
        self.action_changes = 0
        self.lockdown= False
        self.lockdown_count=0
        self.economic_budget_based = True
        self.economic_budget = 1900
        self.capacity_based = True 
        self.reduce_bed = False
        self.reduce_theta = False
        self.limit_changes = False
        self.reduce_compliance = False
        self.compliance_theta = 1
        self.reduce_peak = False
        self.increase_peak_days = False
        self.reduce_total_infected = False
        self.area_reduction = 0.4
        self.total_infected=0
        self.bed_decrease_per_day = (((self.bed_per_1000/1000)-(self.lower_bed_per_1000/1000)) * 12600000)/self.days
        self.theta_decrease_per_day = (self.theta - self.lower_theta)/self.days
        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=100000000, shape=(25,), dtype=np.float32)



        self.use_delta = False
        INPUT =(self.s_0,self.e_0,self.i_0)
        t_start = 0.0; t_end = self.days; t_inc = 1
        t_range = np.arange(t_start, t_end+t_inc, t_inc)
        RES = spi.odeint(self.diff_eqs,INPUT,t_range)
        self.original_infected_list = RES[:,2]


        self.original_area_under_curve = simps(np.array(self.original_infected_list),dx = len(self.original_infected_list))
        

        self.use_delta = True

    def diff_eqs(self,INP,t):  
        '''The main set of equations'''
        Y=np.zeros((3))
        V = INP  
        if self.use_delta:  
            Y[0] = - self.compliance_theta* self.beta*self.delta*self.theta * V[0] * V[2] 
            Y[1] = self.compliance_theta*self.beta*self.delta*self.theta * V[0] * V[2] - self.alpha * V[1] 
            Y[2] = self.alpha * V[1] - self.gamma * V[2] 
        else:
            Y[0] = - self.beta* V[0] * V[2] 
            Y[1] = self.beta* V[0] * V[2] - self.alpha * V[1] 
            Y[2] = self.alpha * V[1] - self.gamma * V[2] 
        return Y   # For odeint

    def _next_observation(self):
        
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
        return_reward=0
        if self.limit_changes:
            '''
            pattern = re.compile(r"0*1*2*3*0*$")
            if bool(re.match(pattern,self.action_string)):
                return 0
            else :
                return -10
            '''

            if self.current_day==0:
                self.previous_day = 0 
            if self.current_day!=self.previous_day:
                return_reward = -10
            self.previous_day=self.current_day
        return return_reward

    def get_difference_in_peak(self):
        original_peak = max(self.original_infected_list)
        current_peak = max(self.infected_list)
        
        return original_peak-current_peak
        
    def get_difference_in_peak_days(self):
        
        original_peak = max(self.original_infected_list)
        current_peak = max(self.infected_list)
        original_peak_day = np.where(self.original_infected_list ==original_peak)[0]
        current_peak_day = np.where(self.infected_list==current_peak)[0]
        difference = current_peak_day - original_peak_day
        return difference[0] 

        
    def reduce_compliance_func(self,action):
        if not self.lockdown and action<3:
            self.lockdown=True
            self.lockdown_count+=1
            self.compliance_theta = self.compliance_theta+ (self.lockdown*10/self.days)
        if action == 3 and self.lockdown:
            self.lockdown=False

    



    def step(self, action):
        action =3
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
        self.reduce_compliance_func(action)

        self.action_string = self.action_string + str(action)

        return_reward = self.limit_changes_func()

        if self.capacity_based:
            if beds_required>self.num_beds:
                self.reward -= 100
                return_reward-= 100
        

        
        
        

        self.reward += self.economy_percent*self.economy
        return_reward+= self.economy_percent*self.economy

        self.rewards.append(self.reward)
        self.current_day +=1

        obs = self._next_observation()
        #if self.current_day == self.days-1:
        if self.current_day>=self.days:
            self.days+=1
        #if self.current_day >=200 and self.i_i*(12600000+1000)<10:
        if self.current_day >= self.days-1:
            #print(max(self.infected_list))
            #print(self.i_i*(12600000+1000))
            
            done = True
            print("Reward",self.reward)
            
        else:
            done= False

        current_infected = (self.infected_list[self.current_day])*self.population
        self.total_infected= (self.infected_list[self.current_day]+self.recovered_list[self.current_day])*self.population
        self.total_cost  +=self.cost*self.cost_percents[action]
        if self.reduce_total_infected :
            self.current_area_under_curve =  simps(np.array(self.infected_list),dx = len(self.infected_list))
            #print()
            
            
            #if self.current_area_under_curve>self.area_reduction*self.original_area_under_curve:
            #if current_infected>self.infection_reduction*self.population:
            if self.total_infected>0.4*self.population:
            #if self.current_area_under_curve>self.area_reduction*self.original_area_under_curve:
                #print("current infected: ",self.current_area_under_curve," original_area_under_curve: ",self.original_area_under_curve)
                
                self.reward-= 100
                return_reward-= 100
                #print("Reward",self.reward)
                #print("Returning ",return_reward)
        if done:
            '''
            infected_percent = self.total_infected/self.population
            
            if infected_percent<=0.4:
                print("inf less")
                self.reward+= (1/(0.4-infected_percent))*1000
                return_reward+=(1/(0.4-infected_percent))*1000
            '''
            print("current infected: ",self.total_infected," original_: ",self.population,"percent infected=",self.total_infected/self.population)

        if done and self.economic_budget_based:
            
            if self.total_cost>  self.economic_budget:
                print("here")
                return_reward-=1000
                self.reward -=1000
            self.reward -= beds_required
            return_reward-= beds_required

        if done and self.reduce_peak:
            difference_in_peak = self.get_difference_in_peak()
            #print("diff",difference_in_peak*1000)
            return_reward +=difference_in_peak*1000
            self.reward +=difference_in_peak*1000

        if done and self.increase_peak_days:
            difference_in_peak_days = self.get_difference_in_peak_days()
            print("diff",difference_in_peak_days*1000)
            return_reward +=difference_in_peak_days*1000
            self.reward +=difference_in_peak_days*1000
        

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
        self.compliance_theta = 1
        self.action_changes = 0
        self.lockdown= False
        self.lockdown_count=0
        self.total_cost=0
        self.total_infected=0
        obs = self._next_observation()
        return obs

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print("Reward = ",self.reward)
        print("Total infected :",self.total_infected/self.population)
        print("Total cost",self.total_cost)
        print(len(self.infected_list))
        print(max(self.infected_list))
        pl.subplot(211)
        if not self.capacity_based:
            pl.plot(self.infected_list, '-g', label='Infected with intervention')
            pl.plot(self.original_infected_list, '-r', label='Infected with no intervention')
        else:
            pl.plot(self.infected_count_list, '-g', label='Infected with intervention')
            #if self.days>len(self.original_infected_list):

            self.original_infected_count_list = [0.05*x*(12600000+1000) for x in self.original_infected_list]
            pl.plot(self.original_infected_count_list, '-r', label='Infected with no intervention')
            if not self.reduce_bed:    
                pl.axhline(y=self.num_beds)
            else:
                axes = pl.gca()
                x_vals = np.array([i for i in range(self.days+1)])
                slope =self.bed_decrease_per_day
                intercept = (self.bed_per_1000/1000) * 12600000
                y_vals = intercept - slope * x_vals
                plt.plot(x_vals, y_vals, '--')

        print(len(self.infected_list))
        print(max(self.infected_list))
        '''
        
        '''

        pl.legend(loc=0)
        pl.xlabel('Time')
        pl.ylabel('Beds Required')

        pl.subplot(212)
        pl.plot(self.action_list, '-r', label='Action')
        pl.xlabel('Time')
        pl.ylabel('Action')
        pl.show()
        return self.action_list