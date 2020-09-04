

import scipy.integrate as spi
import numpy as np
import pylab as pl
import datetime
import pandas as pd

beta=0.1
gamma=0.05
TS=1.0
ND=250
#S0=1-1e-6
#I0=1e-6
S0=0.999
I0=0.001
INPUT = (S0, I0, 0.0)


def diff_eqs(INP,t):  
	'''The main set of equations'''
	Y=np.zeros((3))
	V = INP    
	Y[0] = - beta * V[0] * V[1]
	Y[1] = beta * V[0] * V[1] - gamma * V[1]
	Y[2] = gamma * V[1]
	return Y   # For odeint

t_start = 0.0; t_end = ND; t_inc = TS
t_range = np.arange(t_start, t_end+t_inc, t_inc)
RES = spi.odeint(diff_eqs,INPUT,t_range)

susceptible_list = []
infected_list = []
recovered_list = []
for item in RES:
	susceptible_list.append(item[0])
	infected_list.append(item[1])
	recovered_list.append(item[2])

start = datetime.datetime.strptime("21-06-2019", "%d-%m-%Y")
date_generated = [start + datetime.timedelta(days=x) for x in range(0, ND)]
date_list = []
for date in date_generated:
    date_list.append(date.strftime("%d-%m-%Y"))
df = pd.DataFrame(list(zip(date_list,susceptible_list,infected_list,recovered_list)),columns = ['Date', 'Susceptible', 'Infected','Recovered'])
df.to_csv('data/epidemic_2.csv')


#Ploting
pl.subplot(211)
pl.plot(RES[:,0], '-g', label='Susceptibles')
pl.plot(RES[:,2], '-k', label='Recovereds')
pl.legend(loc=0)
pl.title('Program_2_1.py')
pl.xlabel('Time')
pl.ylabel('Susceptibles and Recovereds')
pl.subplot(212)
pl.plot(RES[:,1], '-r', label='Infectious')
pl.xlabel('Time')
pl.ylabel('Infectious')
pl.show()