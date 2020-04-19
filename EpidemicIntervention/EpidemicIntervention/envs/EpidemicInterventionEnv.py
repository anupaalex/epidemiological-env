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

INIT_ACCOUNT_BALANCE = 100
INIT_AREA_UNDER_CURVE = 0
INIT_PEAK  = 0 
INIT_PEAK_DAY = 0

PRINT = False
MAX_COST_OF_INTERVENTION = 1
MAX_COST_OF_START_INTERVENTION = 1
GAIN_FROM_INTERVENTION_PER_UNIT = 1
TOTAL_POPULATION = 100000
MAX_INTERVENTIONS = 1
MAX_INTERVENTION_DAYS = 10
INTERVENTION_DAYS = 10
class EpidemicInterventionEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(EpidemicInterventionEnv, self).__init__()
        if PRINT:
            print("Init")
        #Loading dataframe
        self.df = pd.read_csv('./data/epidemic_2.csv')
        self.df['Date_new'] = pd.to_datetime(self.df['Date'], format='%d-%m-%Y')
        self.df = self.df.sort_values('Date_new')
        self.initial_df = self.df.copy()
        self.gain = 0
        self.continuous = True #continuos action space or not
        self.beta_const = True #if change in beta should be const
        
        if self.continuous:
            self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4, 7), dtype=np.float16)
        
        self.balance = INIT_ACCOUNT_BALANCE
        self.beta=0.1
        self.gamma=0.05
        self.beta_drop = 0.02
        self.lower_beta = self.beta-self.beta_drop
        self.old_beta = self.beta #old beta to change back to original beta
        
        self.total_interventions = 0

        self.current_step = 0
        self.start_time = -1
        self.end_time = -1
        self.start_outside_intervention = 0
        self.intervention_started = False
        self.intervention_list =[]
        self.max_intervention_reached = False

        # Prices contains the OHCL values for the last five prices
        
        
        global INIT_AREA_UNDER_CURVE
        global INIT_PEAK
        global INIT_PEAK_DAY 
        INIT_AREA_UNDER_CURVE = simps(np.array(self.df['Infected']),dx = len(self.df))
        INIT_PEAK = max(np.array(self.df['Infected']))
        INIT_PEAK_DAY = list(self.df['Infected']).index(max(list(self.df['Infected'])))

        
        #unused
        self.total_intervention_cost = 0
        self.ppl_intervened=0
        self.total_intervention_value = 0
        self.total_intervened_ppl = 0
        

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
        
        obs = np.append(frame, [[
            self.current_step,
            self.current_step,
            self.beta,
            self.gamma,
            self.total_interventions / MAX_INTERVENTIONS,
            self.total_interventions / MAX_INTERVENTIONS,
            self.total_interventions / MAX_INTERVENTIONS,
        ]], axis=0)
        if PRINT:#
            print("returning obs")
        return obs
    def diff_eqs(self,INP,t):  
        
        '''The main set of equations'''
        Y=np.zeros((3))
        V = INP    
        Y[0] = - self.beta_to_use * V[0] * V[1]
        Y[1] = self.beta_to_use * V[0] * V[1] - self.gamma * V[1]
        Y[2] = self.gamma * V[1]
        return Y   # For odeint

    def get_min_area_under_curve_peak_peak_day(self):
        t_range = np.arange(0,len(self.df)+1 , 1.0)
        S0 = self.initial_df.loc[0, "Susceptible"]
        I0 = self.initial_df.loc[0, "Infected"]
        R0 = self.initial_df.loc[0, "Recovered"]
        INPUT = (S0, I0, R0)
        if PRINT:#
            print(INPUT)
        self.beta_to_use = self.lower_beta
        RES = spi.odeint(self.diff_eqs,INPUT,t_range)
        infected_list = []
       
        for item in RES:
            infected_list.append(item[1])
        if PRINT:
            print("max area under curve = ",trapz(infected_list, dx=len(self.df)))
        min_peak = max(infected_list)
        max_peak_day = infected_list.index(min_peak)
        return trapz(infected_list, dx=len(self.df)),min_peak,max_peak_day


    def area_under_curve(self):
        if PRINT:
            print("aread under curve start:%d end : %d"%(self.start_time,self.current_step))
        t_range = np.arange(self.start_time,self.current_step+1 , 1.0)
        S0 = self.df.loc[self.start_time, "Susceptible"]
        I0 = self.df.loc[self.start_time, "Infected"]
        R0 = self.df.loc[self.start_time, "Recovered"]
        INPUT = (S0, I0, R0)
        if PRINT:#
            print(INPUT)
        self.beta_to_use = self.old_beta
        RES_old = spi.odeint(self.diff_eqs,INPUT,t_range)
        infected_list_old = []
        self.beta_to_use = self.beta
        RES_new = spi.odeint(self.diff_eqs,INPUT,t_range)
        infected_list_new = []
        for item in RES_old:
            infected_list_old.append(item[1])
        for item in RES_new:
            infected_list_new.append(item[1])
            
        area_under_curve_before = trapz(infected_list_old, dx=self.current_step-self.start_time+1)
        area_under_curve_after = trapz(infected_list_new, dx=self.current_step-self.start_time+1)
        area_under_curve_diff = area_under_curve_before- area_under_curve_after
        
        if PRINT:#
            print("area area_under_curve_diff = %f "%(area_under_curve_diff))
        return area_under_curve_diff

    def get_infected_list(self):
        infected_list = []

        #from begining to start of intervention
        if PRINT:
            print("From beg to %d"%(self.start_time_calc))
        infected_list.extend(self.df['Infected'][:self.start_time-1])
        #from start to how long th intervention is
        #t_range = np.arange(self.start_time_calc,min(self.start_time+MAX_INTERVENTION_DAYS,len(self.df))+1 , 1.0)
        print("start calc = %d curr +1 %d "%(self.start_time,self.current_step+1))
        t_range = np.arange(self.start_time,self.current_step+1, 1.0)
        S0 = self.df.loc[self.start_time, "Susceptible"]
        I0 = self.df.loc[self.start_time, "Infected"]
        R0 = self.df.loc[self.start_time, "Recovered"]
        print("len1:",len(self.df['Infected'][:self.start_time]))
        INPUT = (S0, I0, R0)
        self.beta_to_use = self.lower_beta
        if PRINT:
            print("From beg intervention  to  end %d %d beta = "%(self.start_time,min(self.start_time+MAX_INTERVENTION_DAYS,len(self.df))),self.beta_to_use)
        RES = spi.odeint(self.diff_eqs,INPUT,t_range)
        for item in RES:
            infected_list.append(item[1])
        print("len2:",len(RES))
        #from end of intervention to end 
       
        #t_range = np.arange(min(self.start_time+MAX_INTERVENTION_DAYS+1,len(self.df))+1 ,len(self.df)+1, 1.0)
        t_range = np.arange(self.current_step+1,len(self.df) , 1.0)
        #k = min(self.start_time+MAX_INTERVENTION_DAYS+1,len(self.df))
        k = min(self.current_step+1,len(self.df)-1)
        S0 = self.df.loc[k, "Susceptible"]
        I0 = self.df.loc[k, "Infected"]
        R0 = self.df.loc[k, "Recovered"]
        INPUT = (RES[-1][0], RES[-1][1], RES[-1][2])
        self.beta_to_use = self.old_beta
        RES = spi.odeint(self.diff_eqs,INPUT,t_range)
        print("len3:",len(RES))
        for item in RES:
            infected_list.append(item[1])
        
        if not PRINT:
            print(len(infected_list))

        return infected_list

    def area_under_curve2(self):
        infected_list = self.get_infected_list()
        return trapz(infected_list, dx=len(infected_list))

        #from end of intervention to rest

    def peak_and_peak_day(self):
        if PRINT:
            print("peak")

        infected_list = self.get_infected_list()
        peak = max(infected_list)
        peak_day = infected_list.index(peak)
        
        return peak,peak_day

    def cost_start_intervention(self):
        if self.start_time ==self.current_step:
            #cost = (self.total_interventions+1/MAX_INTERVENTIONS) * MAX_COST_OF_START_INTERVENTION
            #cost = (MAX_INTERVENTIONS/self.total_interventions) * MAX_COST_OF_START_INTERVENTION
            cost = 10
        else:
            cost = 0
        if PRINT:
            print("cost_start_intervention=",cost)
        return cost

    def cost_of_intervention(self):
        
        if self.start_time!=-1:
            cost = (self.current_step-self.start_time)  * MAX_COST_OF_INTERVENTION            
        else:
            cost = 0
        if PRINT:
            print("cost_of_intervention=",cost)
        return cost



    def gain_from_intervention(self):
        if self.start_time!=-1 and self.start_time<=self.current_step:
            #if self.start_time!=-1:
            area_under_curve = self.area_under_curve2()
            min_area,min_peak,max_peak_day = self.get_min_area_under_curve_peak_peak_day()
            max_area_under_curve_diff = INIT_AREA_UNDER_CURVE - min_area
            area_gain = ((INIT_AREA_UNDER_CURVE - area_under_curve)/max_area_under_curve_diff) * GAIN_FROM_INTERVENTION_PER_UNIT

            peak,peak_day = self.peak_and_peak_day()
            peak_gain = (INIT_PEAK - peak) /(INIT_PEAK-min_peak)
            #peak_day_gain = (peak_day - INIT_PEAK_DAY)/(max_peak_day-INIT_PEAK_DAY)
            peak_day_gain = (peak_day - INIT_PEAK_DAY)
            print("All max area =%f peak =%f peak day = %f "%(min_area,INIT_PEAK-min_peak,max_peak_day-INIT_PEAK_DAY))
            print("All actual area =%f peak =%f peak day = %f"%(INIT_AREA_UNDER_CURVE - area_under_curve,INIT_PEAK - peak, peak_day - INIT_PEAK_DAY))
            print("***************Gains : area =%f peak =%f peak day = %f"%(area_gain,peak_gain,peak_day_gain))
            gain = 0*area_gain + 0*peak_gain + peak_day_gain
        else:
            gain = 0
        if PRINT:
            print("gain_from_intervention=",cost)
        return gain



    def _take_action(self, action):
        self.stopped = False

        print(action)
        current_cost = self.df.loc[self.current_step, "Infected"] * self.beta 
        if PRINT:#
            print("current step, price",self.current_step,current_cost)
        
        action_type = action[0]
        beta_percent_red = action[1]
        if action_type <0.5 and self.total_interventions < MAX_INTERVENTIONS :
            if not PRINT:#
                print("1*********start intervention")
            

            #for before the intervention starts
            if not self.intervention_started:
                t_range = np.arange(self.start_outside_intervention, len(self.df['Susceptible']), 1.0)
                S0 = self.df.loc[self.start_outside_intervention, "Susceptible"]
                I0 = self.df.loc[self.start_outside_intervention, "Infected"]
                R0 = self.df.loc[self.start_outside_intervention, "Recovered"]
                INPUT = (S0, I0, R0)
                self.beta_to_use = self.old_beta
                RES = spi.odeint(self.diff_eqs,INPUT,t_range)
                susceptible_list = []
                infected_list = []
                recovered_list = []
                for item in RES:
                    susceptible_list.append(item[0])
                    infected_list.append(item[1])
                    recovered_list.append(item[2])
                if len(infected_list) >0:
                    self.df.loc[self.start_outside_intervention: len(self.df['Susceptible'])-1, 'Susceptible'] = susceptible_list
                    self.df.loc[self.start_outside_intervention: len(self.df['Susceptible'])-1, 'Infected'] = infected_list
                    self.df.loc[self.start_outside_intervention: len(self.df['Susceptible'])-1, 'Recovered'] = recovered_list
                
           
            #to be considered later
            #total_possible = self.balance/current_cost
            #prev_cost = self.cost_basis * self.ppl_intervened
            #additional_cost = ppl_intervened_new * current_cost
            #self.balance -= additional_cost
            #self.cost_basis = (
            #    prev_cost + additional_cost) / (self.ppl_intervened + ppl_intervened_new)
            #self.total_interventions +=1

            if self.start_time==-1:
                self.old_beta = self.beta
                if PRINT:
                    print("setting new start time")
                if self.beta_const :
                    self.beta = self.beta - self.beta_drop
                else:
                    self.beta = self.beta - beta_percent_red* self.beta
                self.intervention_started = True
                self.start_time = self.current_step
                self.intervention_list.append((self.start_time))
        else:
            if not PRINT:
                print("no action")
        if self.start_time!=-1 and self.current_step == self.start_time+INTERVENTION_DAYS:
            if not PRINT:#
                print("2*********stop intervention")

            #to be considered later
            # Stop intervention on amount % of people
            #gain_per_person = self.df.loc[self.current_step, "Infected"] * self.beta 
            #self.balance += ppl_intervened_stop * gain_per_person
            #self.total_intervened_ppl += ppl_intervened_stop
            #intervention value can be changed to diff in area under curve
            #self.total_intervention_value += ppl_intervened_stop * gain_per_person

            #if intervention has been started
            if self.start_time!=-1:
                #self.gain = self.gain_from_intervention()
                self.end_time = self.current_step
                
                t_range = np.arange(self.start_time, len(self.df['Susceptible']), 1.0)
                S0 = self.df.loc[self.start_time, "Susceptible"]
                I0 = self.df.loc[self.start_time, "Infected"]
                R0 = self.df.loc[self.start_time, "Recovered"]
                INPUT = (S0, I0, R0)
                self.beta_to_use = self.beta
                RES = spi.odeint(self.diff_eqs,INPUT,t_range)
                susceptible_list = []
                infected_list = []
                recovered_list = []
                for item in RES:
                    susceptible_list.append(item[0])
                    infected_list.append(item[1])
                    recovered_list.append(item[2])
                self.intervention_list.append((self.start_time,self.end_time,self.beta_to_use))
                if len(infected_list) >0:
                    self.df.loc[self.start_time: len(self.df['Susceptible'])-1, 'Susceptible'] = susceptible_list
                    self.df.loc[self.start_time: len(self.df['Susceptible'])-1, 'Infected'] = infected_list
                    self.df.loc[self.start_time: len(self.df['Susceptible'])-1, 'Recovered'] = recovered_list
                   
                #
                self.start_time =-1
                self.start_outside_intervention = self.current_step+1
                self.intervention_started = False
                self.beta = self.old_beta
                self.total_interventions +=1
        
        
        #if all the interventions are done the rest have to be filled    
        if self.total_interventions==MAX_INTERVENTIONS and not self.max_intervention_reached:
            if PRINT:#
                print("setting rest to old val************&************")
            self.max_intervention_reached = True
            t_range = np.arange(self.current_step, len(self.df['Susceptible']), 1.0)
            S0 = self.df.loc[self.current_step, "Susceptible"]
            I0 = self.df.loc[self.current_step, "Infected"]
            R0 = self.df.loc[self.current_step, "Recovered"]
            INPUT = (S0, I0, R0)
            self.beta_to_use = self.old_beta
            if PRINT:#
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!interven Diff eq***************************** current step == %d beta = %f start = %d S,I,R =(%f,%f,%f)"%(self.current_step,self.beta_to_use,self.current_step,S0,I0,R0))
            RES = spi.odeint(self.diff_eqs,INPUT,t_range)
            if PRINT:#
                print(RES)
            susceptible_list = []
            infected_list = []
            recovered_list = []
            for item in RES:
                susceptible_list.append(item[0])
                infected_list.append(item[1])
                recovered_list.append(item[2])
            if PRINT:#
                print("len list", len(susceptible_list))
            if len(infected_list) >0:
                self.df.loc[self.current_step: len(self.df['Susceptible'])-1, 'Susceptible'] = susceptible_list
                self.df.loc[self.current_step: len(self.df['Susceptible'])-1, 'Infected'] = infected_list
                self.df.loc[self.current_step: len(self.df['Susceptible'])-1, 'Recovered'] = recovered_list


            


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        
        done = False
        if self.current_step > len(self.df.loc[:, 'Susceptible'].values) - 8:
            #if itervention was started but hasn't ended yet
            if self.start_time!=-1:
                self.intervention_list.append((self.start_time,self.current_step))
                t_range = np.arange(self.start_time, len(self.df['Susceptible']), 1.0)
                S0 = self.df.loc[self.start_time, "Susceptible"]
                I0 = self.df.loc[self.start_time, "Infected"]
                R0 = self.df.loc[self.start_time, "Recovered"]
                INPUT = (S0, I0, R0)
                self.beta_to_use = self.lower_beta
                RES = spi.odeint(self.diff_eqs,INPUT,t_range)
                susceptible_list = []
                infected_list = []
                recovered_list = []
                for item in RES:
                    susceptible_list.append(item[0])
                    infected_list.append(item[1])
                    recovered_list.append(item[2])
                self.intervention_list.append((self.start_time,self.end_time,self.beta_to_use))
                if len(infected_list) >0:
                    self.df.loc[self.start_time: len(self.df['Susceptible'])-1, 'Susceptible'] = susceptible_list
                    self.df.loc[self.start_time: len(self.df['Susceptible'])-1, 'Infected'] = infected_list
                    self.df.loc[self.start_time: len(self.df['Susceptible'])-1, 'Recovered'] = recovered_list
            if PRINT:#
                print("Setting current step back")
            self.current_step = random.randint(
            0, len(self.df.loc[:, 'Susceptible'].values) - 8)

            #done = True
        
        done = False
        reward_finish = 0 
        if self.total_interventions==MAX_INTERVENTIONS:
            if PRINT:#
                print("Setting current step back")
            self.current_step = 0
            done = True
            print(self.intervention_list)
            new_peak_day = list(self.df['Infected']).index(max(list(self.df['Infected'])))
            reward_finish = new_peak_day - INIT_PEAK_DAY 
            print("Finish reward",reward_finish)




        obs = self._next_observation()
        cost_intervention = self.cost_of_intervention()
        cost_start = self.cost_start_intervention()
        gain_intervention = self.gain_from_intervention()
        WCI=1
        WCS=1
        WG=1
        WRF=0
        reward = -(WCI*cost_intervention + WCS*cost_start - WG*gain_intervention - WRF* reward_finish) 

        #test
        #if action[0]<1:
         #   reward -= 10

        if not PRINT:
            print("Current step = %d start_time = %d Reward %f cost of start = %f cost of continueing = %f gain from int = %f gain from finish reward = %d"%(self.current_step,self.start_time,reward,WCS*cost_start, WCI*cost_intervention,WG* gain_intervention,WRF* reward_finish))
        self.current_step += 1

        return obs, reward, done, {}

    def set_step(self):
        print("Resetting current_step")
        self.current_step = 0

    def reset(self):
        if PRINT:
            print("reset**************************")
        # Reset the state of the environment to an initial state
        
        self.area_under_curve_diff = 0
        self.total_interventions = 0
        self.beta=0.1
        self.gamma=0.05
        self.gain = 0
        self.df = pd.read_csv('./data/epidemic_2.csv')
        self.df['Date_new'] = pd.to_datetime(self.df['Date'], format='%d-%m-%Y')
        self.df = self.df.sort_values('Date_new')
        
        #Set the current step to a random point within the data frame
        #self.current_step = random.randint(
         #   0, len(self.df.loc[:, 'Susceptible'].values) - 8)
        self.total_interventions=0
        self.current_step = 0
        self.start_time = -1
        self.end_time = -1
        self.start_outside_intervention = 0
        self.intervention_started = False
        self.intervention_list=[]
        self.max_intervention_reached = False
        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit =  self.total_intervention_value
        new_peak = max(np.array(self.df['Infected']))
        
        print("Old peak value : %f New peak value : %f Peak moved by : %f"%(INIT_PEAK,new_peak,INIT_PEAK-new_peak))
        
        new_peak_day = list(self.df['Infected']).index(max(list(self.df['Infected'])))
        print("Old peak day : %f New peak day : %f Peak day moved by : %f"%(INIT_PEAK_DAY,new_peak_day,INIT_PEAK_DAY-new_peak_day))

        new_area_under_curve = trapz(list(self.df['Infected']), dx=len(self.df['Infected']))
        print("Old area under curve : %f New area under curve : %f Diff in area under curve : %f"%(INIT_AREA_UNDER_CURVE,new_area_under_curve,INIT_AREA_UNDER_CURVE-new_area_under_curve))
        
        print("Intervention list : ",self.intervention_list)
        return self.df
