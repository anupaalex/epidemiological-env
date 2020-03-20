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
MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000
MAX_INTERVENTION_COST = 100
MAX_NUM_INTERVENTIONS = 100
INITIAL_ACCOUNT_BALANCE = 100
#INIT_AREA_UNDER_CURVE=100000
TOTAL_POPULATION = 100000

#INITIAL_ACCOUNT_BALANCE = 10000


class EpidemicInterventionEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(EpidemicInterventionEnv, self).__init__()

        self.df = df
        print(self.df.head())
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format start intervention x%, stop x%, no op, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.beta=0.3
        self.cost_basis = 0
        self.gamma=0.14286
        self.total_intervention_cost = 0
        self.total_intervention_value = 0
        self.ppl_intervened=0
        self.total_intervened_ppl = 0
        self.total_interventions = 0
        self.current_step = 0
        self.start_time = None
        self.end_time = None
        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4, 7), dtype=np.float16)
        global INIT_AREA_UNDER_CURVE
        INIT_AREA_UNDER_CURVE = simps(np.array(self.df['Infected']),dx = len(self.df))
        

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                        6, 'Susceptible'].values/ TOTAL_POPULATION,
            self.df.loc[self.current_step: self.current_step +
                        6, 'Infected'].values/ TOTAL_POPULATION,
            self.df.loc[self.current_step: self.current_step +
                        6, 'Recovered'].values/ TOTAL_POPULATION 

        ])
        
        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.ppl_intervened/100 ,
            self.beta,
            self.gamma,
            self.cost_basis / MAX_INTERVENTION_COST,
            self.total_intervened_ppl / 100,
            self.total_intervention_cost / (MAX_NUM_INTERVENTIONS * MAX_INTERVENTION_COST)
        ]], axis=0)
        print("returning obs")
        return obs
    def diff_eqs(self,INP,t):  
        '''The main set of equations'''
        Y=np.zeros((3))
        V = INP    
        Y[0] = - self.beta * V[0] * V[1]
        Y[1] = self.beta * V[0] * V[1] - self.gamma * V[1]
        Y[2] = self.gamma * V[1]
        return Y   # For odeint

    def _take_action(self, action):

        print("current price",current_cost)
        action_type = action[0]
        amount = action[1]

        if action_type >1:
            print("1")
            # Start intervention on amount % of people
            total_possible = self.balance/current_cost
            ppl_intervened_new = total_possible * amount
            self.beta = self.beta - (0.01)
            print("beta",self.beta)
            prev_cost = self.cost_basis * self.ppl_intervened
            additional_cost = ppl_intervened_new * current_cost

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.ppl_intervened + ppl_intervened_new)
            self.ppl_intervened += ppl_intervened_newrfvf
            self.total_interventions +=1
            print("People intervened",self.ppl_intervened)
            self.start_time = self.current_step
        elif action_type >0.5:
            print("2*********")
            # Stop intervention on amount % of people
            gain_per_person = self.df.loc[self.current_step, "Infected"] * self.beta 
            ppl_intervened_stop = self.ppl_intervened * amount
            self.balance += ppl_intervened_stop * gain_per_person
            self.ppl_intervened -= ppl_intervened_stop
            self.total_intervened_ppl += ppl_intervened_stop
            #intervention value can be changed to diff in area under curve
            self.total_intervention_value += ppl_intervened_stop * gain_per_person

            if self.start_time!=None:

                self.end_time = self.current_step

                t_range = np.arange(self.start_time, self.end_time+1, 1.0)
                S0 = self.df.loc[self.start_time, "Susceptible"]
                I0 = self.df.loc[self.start_time, "Infected"]
                R0 = self.df.loc[self.start_time, "Recovered"]
                INPUT = (S0, I0, R0)
                print(INPUT)
                RES = spi.odeint(self.diff_eqs,INPUT,t_range)
                print(RES)
                susceptible_list = []
                infected_list = []
                recovered_list = []
                for item in RES:
                    susceptible_list.append(item[0])
                    infected_list.append(item[1])
                    recovered_list.append(item[2])
                print("len list", len(susceptible_list))
                print("len df",len(self.df.loc[self.start_time: self.end_time, 'Susceptible']))
                print("Before : ",self.df.loc[self.start_time: self.end_time, 'Susceptible'].values)
                if len(infected_list) >0:
                    self.df.loc[self.start_time: self.end_time, 'Susceptible'] = susceptible_list
                    self.df.loc[self.start_time: self.end_time, 'Infected'] = infected_list
                    self.df.loc[self.start_time: self.end_time, 'Recovered'] = recovered_list
                   
                print("After :",self.df.loc[self.start_time: self.end_time, 'Susceptible'].values)
                print("2$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$",self.start_time,self.end_time)
                self.start_time =None
        else:
            print("********************************************************")
            
        if self.ppl_intervened == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Susceptible'].values) - 8:
            print("Setting current step back")
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        #reward = self.balance * delay_modifier
        reward = 0
        done = self.total_intervention_value <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        
        #self.area_under_curve_diff = 0
        #self.intervention_head_count =0
        #self.cost_basis = 0
        #self.total_interventions = 0
        #self.total_intervention_cost= 0
        #self.beta=0.3
        #self.gamma=0.14286
        #self.total_intervention_cost = 0
        #self.total_intervened_ppl = 0
        #self.total_intervention_value = 0
        #self.ppl_intervened=0
        #self.balance = INITIAL_ACCOUNT_BALANCE
        # Set the current step to a random point within the data frame
        #self.current_step = random.randint(
        #    0, len(self.df.loc[:, 'Susceptible'].values) - 8)
        #self.current_step = 0
        print("current_step======",self.current_step)
        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit =  self.total_intervention_value

        print(f'Step: {self.current_step}')
        print(f"Beta:",self.beta)
        #print(f'Balance: {self.balance}')
        print(
            f'Total intervened people now : {self.total_intervened_ppl} (Total interventions: {self.total_interventions})')
        print(
            f'Avg cost for started intervention: {self.cost_basis} (Total intervention cost: {self.total_intervention_cost})')
        #print(
        #    f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Gain due to interventions: {profit}')
        return self.df
