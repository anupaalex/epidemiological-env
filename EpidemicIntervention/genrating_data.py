import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import math
num_days = 200
start_day = pd.datetime.today()
total_population = 100000
mult_factor = total_population/100
date_list = []
susceptible_list = []
infected_list = []
recovered_list = []
infected = 0 
susceptible = total_population
recovered = 0
max_infected = 0.85 *total_population
num_days_inc = int(num_days/2)
num_days_dec = num_days -num_days_inc

#y = a*b^x
y = total_population
a = mult_factor
x = num_days_inc
b = math.exp(math.log(y/a)/x)

print(b)

#date_list = pd.date_range(start_day, periods=num_days).tolist()
#print(date_list)

start = datetime.datetime.strptime("21-06-2019", "%d-%m-%Y")
date_generated = [start + datetime.timedelta(days=x) for x in range(0, num_days)]

for date in date_generated:
    date_list.append(date.strftime("%d-%m-%Y"))





###################for increasing
infected_list.append(infected)
susceptible_list.append(susceptible)
recovered_list.append(recovered)

for i in range(0,num_days_inc):
	new_infected = min(int(mult_factor*(b**i)),max_infected)
	infected_list.append(new_infected)
	susceptible_list.append(total_population-new_infected)
	recovered_list.append(0)
	susceptible = total_population-new_infected

###################for decreasing
#temp_infected = []

#infected = infected_list[-1]
#for i in range(0,num_days_inc):




dec_infected_list = sorted(infected_list,reverse = True)
infected = dec_infected_list[0]
infected_total = dec_infected_list[0]
for i in range(0,num_days_dec):
	new_infected = dec_infected_list[i]
	recovered = infected_total-new_infected

	infected_list.append(new_infected)
	susceptible_list.append(susceptible)
	recovered_list.append(recovered)

	infected = dec_infected_list[i]

df = pd.DataFrame(list(zip(date_list,susceptible_list,infected_list,recovered_list)),columns = ['Date', 'Susceptible', 'Infected','Recovered'])
df.to_csv('data/epidemic_1.csv')

plt.plot(infected_list)
plt.show()
