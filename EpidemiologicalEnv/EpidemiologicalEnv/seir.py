#!/usr/bin/env python

####################################################################
###    This is the PYTHON version of program 2.6 from page 41 of   #
### "Modeling Infectious Disease in humans and animals"            #
### by Keeling & Rohani.										   #
###																   #
### It is the SEIR epidemic with equal births and deaths.          #
### Note we no-longer explicitly model the recovered class.	       #
####################################################################

###################################
### Written by Ilias Soumpasis    #
### ilias.soumpasis@ucd.ie (work) #
### ilias.soumpasis@gmail.com	  #
###################################

import scipy.integrate as spi
import numpy as np
import pylab as pl

mu=1/(70*365.0)
beta=0.4482
sigma=0.1923
gamma=0.1724
ND=200
TS=1

S0=12600000/(12600000+1000)
E0=1000/(12600000+1000)
I0=0
INPUT = (S0, E0, I0)

def diff_eqs(INP,t):  
	'''The main set of equations'''
	Y=np.zeros((3))
	V = INP    
	Y[0] =  - beta * V[0] * V[2] 
	Y[1] = beta * V[0] * V[2] - sigma * V[1]
	Y[2] = sigma * V[1] - gamma * V[2]
	return Y   # For odeint



t_start = 0.0; t_end = ND; t_inc = TS
t_range = np.arange(t_start, t_end+t_inc, t_inc)
RES = spi.odeint(diff_eqs,INPUT,t_range)

Rec=1. - (RES[:,0]+RES[:,1]+RES[:,2])
print(RES)

#Ploting
#pl.subplot(311)
pl.plot(RES[:,0], '-g', label='Susceptibles')
#pl.title('Program_2_6.py')
pl.xlabel('Time')
pl.ylabel('Fraction of population')

pl.plot(RES[:,1], '-m', label='Exposed')
pl.plot(RES[:,2], '-r', label='Infectious')
pl.plot(Rec, '-k', label='Recovereds')
pl.legend(loc=0)



pl.show()