# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:51:41 2019

@author: jschmidt
"""

import time

def do_something_fancy():
    time.sleep(0.3)
    
t0 = time.time()

do_something_fancy()

elapsed = time.time() - t0

print(elapsed)

%timeit 39+2

from contexttimer import timer, Timer

with Timer() as t:
    sleep(1)
    
t.elpased

import numpy as np
a = np.arange(8)
a
a.shape
a.reshape(2, 4)

a.reshape(2, 4).ndim

5 * np.ones((3,3))

a = np.arange(8).reshape(2,4)


# convert R-Code
import pandas as pd
import matplotlib.pyplot as plt

tab_emissions_costs = pd.read_csv("data/results_ubrmskript.csv")
tab_emissions_costs["Scenario"] = tab_emissions_costs["Scenario"].str.strip()
tab_emissions_costs.rename(columns={'Scenario':'Szenario'}, inplace=True)

idcs_scenarios = ((tab_emissions_costs["Szenario"] == "EUA160") | 
                  (tab_emissions_costs["Szenario"] == "s8")) 

points_scenarios = tab_emissions_costs[idcs_scenarios]

erc_colors = ["#C72321","#6E9B9E"]

tab_emissions_costs = tab_emissions_costs.sort_values("Emissions")              

f, ax = plt.subplots(1)
ax.plot(tab_emissions_costs["Emissions"], 
                            tab_emissions_costs["Effective Cost"], "o-")
ax.plot(points_scenarios["Emissions"],
                      points_scenarios["Effective Cost"], "o")    
              
plt.legend()