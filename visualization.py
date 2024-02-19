# -*- coding: utf-8 -*-

#%% === Importing packages ===
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from tqdm.notebook import tqdm
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
import plotly.express as px
from scipy.fft import fft
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import pyomo
import inspect 
import itertools 
import logging 
import math
import weakref
import xarray
import pypsa

from pyomo.environ import *
from pypsa.descriptors import (
    get_switchable_as_dense,
    allocate_series_dataframes
)
from pypsa.opt import (
    LConstraint,
    LExpression,
    free_pyomo_initializers,
    l_constraint,
)

from datetime import datetime 
from itertools import product
from DatenSmall.Zeitreihe import ladeVorverarbeiteteDaten #JOH

#numeric limit to 0 and to infinity.
bigM = 1e6
smallM = 1e-6 

#%% === load dfins ===
inputFile = r'InputData\Parameter_Input_2023.csv'
dfins = pd.read_csv(inputFile, index_col=0)

#%% === functions for reading simulation results ===
def getVariationInput(name_varia, i_varia, dfins, name_folder = None, path = None):
    '''
    inputs: dfins is the original dataframe, i_var is the iteration of the variation named name_varia we are looking at (i from 0 to N-1) 
    outputs: the updated dataframe, a pricefactor for DA price scaling, and the total number of variations considered in this variation
    '''
    dfins = dfins.copy(deep=True) #will be used as inputs for the 
    priceFactor = -1 
    
    if name_varia == 'varyInvest_X':
        # varying a factor that is multiplied with all investment costs
        x_axis = np.arange(0.25,2.5+1e-3,0.125)*100
        N_varia = len(x_axis)
        x_label = 'Investment Cost Multiplier'
        x_unit = '[%]'
        dfins['capcost'] = dfins['capcost'] * x_axis[i_varia] / 100

    if name_varia == 'varyH2_X':
        # varying a factor that is multiplied with all investment costs
        x_axis = np.arange(0,300+1e-3,12.5)
        N_varia = len(x_axis)
        x_label = 'Hydrogen Price'
        x_unit = '[EUR/MWh]'
        dfins.at['H2_HP_market','margcost'] = x_axis[i_varia]
        
    if name_varia == 'varyCH4_X':
        # varying a factor that is multiplied with all investment costs
        x_axis = np.arange(0,300+1e-3,12.5)
        N_varia = len(x_axis)
        x_label = 'Methane Price'
        x_unit = '[EUR/MWh]'
        dfins.at['CH4market','margcost'] = x_axis[i_varia]
            
    if name_varia == 'varyCH3OH_X':
        # varying a factor that is multiplied with all investment costs
        x_axis = np.arange(0,300+1e-3,12.5)
        N_varia = len(x_axis)
        x_label = 'Methanol Price'
        x_unit = '[EUR/MWh]'
        dfins.at['CH3OHmarket','margcost'] = x_axis[i_varia]
        
    if name_varia == 'varyCO2certificate_X':
        # varying a factor that is multiplied with all investment costs
        x_axis = np.arange(0,300+1e-3,12.5)
        N_varia = len(x_axis)
        x_label = 'CO2 Certificate Price'
        x_unit = '[EUR/MWh]'
        dfins.at['CO2_exhaust','margcost'] = -x_axis[i_varia] *(1-dfins.at['CHP','x_biogen'])      

    if name_varia == 'varyPower_X':
        x_axis = np.hstack([np.arange(25,150,12.5),np.arange(150,300+1e-6,25)])
        N_varia = len(x_axis)
        x_label = 'Average Day-Ahead Price'
        x_unit = '[EUR/MWh]'
        priceFactor = x_axis[i_varia] 
          
    if name_varia == 'varyMulti_X':
        #vary all conditions simultaneously to affect H2, CC or electric heating positive or negatively. Worst-case analysis so to speak
        x_axis = ['Ref.', 'H2-20', 'H2-15','H2-10', 'H2-5', 'H2+5', 'H2+10', 'H2+15', 'H2+20',
                  'CC-20', 'CC-15', 'CC-10', 'CC-5', 'CC+5', 'CC+10', 'CC+15', 'CC+20',
                  'EH-20', 'EH-15', 'EH-10', 'EH-5', 'EH+5', 'EH+10', 'EH+15', 'EH+20']
        signs = {'H2':{'invest':-1, 'H2':1, 'CH4':1, 'CH3OH':1, 'CO2ETS':-1, 'DA':-1}, #optimistic electrolyser
                'CC':{'invest':-1, 'H2':-1, 'CH4':-1, 'CH3OH':1, 'CO2ETS':1, 'DA':-1}, #optimistic Carbon Capture
                'EH':{'invest':1, 'H2':-1, 'CH4':0, 'CH3OH':0, 'CO2ETS':1, 'DA':1}, #optimistic electric heater
                }
        N_varia = len(x_axis)
        x_label = 'Pessimistic and optimistic scenarios for different technologies'
        x_unit = ''
        if i_varia != 0:
            var = x_axis[i_varia][0:2]
            change = float(x_axis[i_varia][2:])/100
            dfins['capcost'] *= (1+signs[var]['invest']*change)
            dfins.at['H2_HP_market','margcost'] *= (1+signs[var]['H2']*change)
            dfins.at['CH4market','margcost'] *= (1+signs[var]['CH4']*change)
            dfins.at['CH3OHmarket','margcost'] *= (1+signs[var]['CH3OH']*change)
            dfins.at['CO2_exhaust','margcost'] *= (1+signs[var]['CO2ETS']*change)
            priceFactor = 95*(1+signs[var]['DA']*change)
        
    # store description to the folder where all results will be stored
    if name_folder is not None:
        combined_path = os.path.join(path or '', name_folder)
        # Check if the directory exists, and create it if not
        if not os.path.exists(combined_path):
            os.makedirs(combined_path)
        # File path to save the pickled data
        file_path = os.path.join(combined_path, f'description.pkl')
        # Dump the data to the file using pickle.dump and file open
        with open(file_path, 'wb') as file:
            pickle.dump([name, x_axis, x_label, x_unit], file)
    return dfins, priceFactor, N_varia

def getVariationOutput(name_folder, i_varia, N_varia, path=None):
    '''get the results of the variation'''
#    print('TO DO')
    # Combine path
    combined_path = os.path.join(path or '', name_folder)
    # Check if the directory exists, and create it if not
    if not os.path.exists(combined_path):
        os.makedirs(combined_path)
    # Export the network
    network = pypsa.Network()
    network.import_from_hdf5(os.path.join(combined_path, f'it_{i_varia}_{N_varia}.hdf5'))
    return network
def getVariationSetup(name_folder, path = None):
        combined_path = os.path.join(path or '', name_folder)
        file_path = os.path.join(combined_path, f'description.pkl')
        # Dump the data to the file using pickle.dump and file open
        with open(file_path, 'rb') as file:
            name, x_axis, x_label, x_unit = pickle.load(file)
        return (name, x_axis, x_label, x_unit)

#%% === setup for plots ===
labels = {
    'heatrod-S0000': 'Electric Heater',
    'heatpump_60_120-S0000': 'Heat Pump (60-120)', 
    'H2compression-S0000': 'H$_2$-Compression',
    'electrolyzer-S0000': 'Electrolyzer',
    'CH3OHcatalysis-S0000': 'CH$_3$OH-Synthesis',
    'CH4catalysis-S0000': 'CH$_4$-Synthesis',
    'CHP_CC-S0000': 'Carbon Capture Unit',
    'CO2liquidification-S0000': 'CO$_2$-Liquification',
    'O2compression-S0000': 'O$_2$-Compression',
    'heatstore-S0000': 'Heat Storage Capacity',
    'heatcharge-S0000': 'Heat Storage Power',
    'dagensell-S0000': 'DA Sales',
    'dagenbuy-S0000': 'DA Purchase',
    'H2_HP_market-S0000': 'H$_2$',
    'CH3OH-S0000': 'CH$_3$OH',
    'CH4-S0000': 'CH$_4$',
    'CO2_L_market-S0000': 'CO$_2$ ',
    'O2_HP_market-S0000': 'O$_2$',
    'CHP_exhaust-S0000': 'CO$_2$ Emissions',
    'batcharge-S0000': 'BESS Power',
    'batstore-S0000': 'BESS Capacity',
}

#https://matplotlib.org/stable/gallery/color/named_colors.html
colors = {
    'heatrod-S0000': 'red',
    'heatpump_60_120-S0000': 'brown', 
    'H2compression-S0000': 'darkorange',
    'electrolyzer-S0000': 'khaki',
    'CH3OHcatalysis-S0000': 'forestgreen',
    'CH4catalysis-S0000': 'darkturquoise', #teal
    'CHP_CC-S0000': 'k',
    'CO2liquidification-S0000': 'darkgrey',
    'O2compression-S0000': 'darkgoldenrod',
    'heatstore-S0000': 'violet',
    'heatcharge-S0000': 'brown',
    'dagensell-S0000': 'm',
    'dagenbuy-S0000': 'plum',
    'H2_HP_market-S0000': 'khaki',
    'CH3OH-S0000': 'forestgreen',
    'CH4-S0000': 'darkturquoise',
    'CO2_L_market-S0000': 'lightgrey',
    'O2_HP_market-S0000': 'darkgoldenrod',
    'CHP_exhaust-S0000': 'slategray',
    'batcharge-S0000': 'darkblue',
    'batstore-S0000': 'royalblue',
    
}

edgecolors = {
    'electrolyzer-S0000': 'white',
    'CH3OHcatalysis-S0000': 'white',
    'CH4catalysis-S0000': 'white',
    'H2_HP_market-S0000':'white',
    'CH3OH-S0000': 'white',
    'CH4-S0000': 'white',
    'heatrod-S0000':'white',
}

hatches = {
    'electrolyzer-S0000': '//',
    'CH3OHcatalysis-S0000': 'xx',
    'CH4catalysis-S0000': '..', 
    'H2_HP_market-S0000': '//',
    'CH3OH-S0000': 'xx',
    'CH4-S0000': '..',
    'heatrod-S0000':'\\',
}

markers = {
    'heatrod-S0000': '<',
    'heatpump_60_120-S0000': '>', 
    'H2compression-S0000': '2',
    'electrolyzer-S0000': '^',
    'CH3OHcatalysis-S0000': 's',
    'CH4catalysis-S0000': 'p',
    'CHP_CC-S0000': 'o',
    'CO2liquidification-S0000': 'd',
    'O2compression-S0000': '1',
    'heatstore-S0000':'D',
    'heatcharge-S0000':'o',
    'batstore-S0000':'s',
    'batcharge-S0000':'o',
}

markersizes = {
    'heatrod-S0000': 5,
    'heatpump_60_120-S0000': 5, 
    'H2compression-S0000': 7,
    'electrolyzer-S0000': 8,
    'CH3OHcatalysis-S0000': 6,
    'CH4catalysis-S0000': 6,
    'CHP_CC-S0000': 3,
    'CO2liquidification-S0000': 5,
    'O2compression-S0000': 7,
    'heatstore-S0000':5,
    'heatcharge-S0000':5,
    'batstore-S0000':5,
    'batcharge-S0000':3,
}

# == helper function to deal with edges of fill_between ==
def fillEdges(y,y_prev):
    '''function that creates a small offset to correctly treat edges of fill_between'''
    eps = 1e-6
    n = len(y)
    updated_y = (np.copy(y).astype(float))
    for i in range(1, n - 1):
        if y[i] == y_prev[i] and (y[i - 1] != y_prev[i - 1] or y[i + 1] != y_prev[i + 1]):
            updated_y[i] = updated_y[i] + eps

    if n > 1 and y[0] == y_prev[0] and y[1] != y_prev[1]:
        updated_y[0] += eps

    if n > 1 and y[n - 1] == y_prev[n - 1] and y[n - 2] != y_prev[n - 2]:
        updated_y[n - 1] += eps

    return updated_y

x_reference = {
    'varyCH4':dfins.at['CH4market','margcost'],
    'varyCH3OH':dfins.at['CH3OHmarket','margcost'],
    'varyCO2certificate':-dfins.at['CO2_exhaust','margcost']/(1-dfins.at['CHP','x_biogen']),
    'varyInvest':100,
    'varyPower':95.175452,
    'varyH2':dfins.at['H2_HP_market','margcost'],
    'testPower':95.175452,
}
x_lims = {
    'varyCH4':(0,300),
    'varyCH3OH':(0,300),
    'varyCO2certificate':(0,300),
    'varyInvest':(25,250),
    'varyPower':(25,300),
    'varyH2':(0,300),
    'testPower':(25,300),
}
referenceProfits = 11652850.54147335

#%% === plotting parameter variation ===
# == load results ==
def readResults(name_folder, path):
    name_varia = '_'.join(name_folder.split('_')[:-2])
    
    _, __, N_varia = getVariationInput(name_varia, 0, dfins)
    
    #load simulation setup
    name_simulation, x_axis, x_label, x_unit = getVariationSetup(name_folder, path)
    optNoms = {}
    optFullLoad = {}
    soldAmounts = {}
    profits = {}
    for i_varia in tqdm(range(N_varia)):
        #load simulation result i_varia
        n = getVariationOutput(name_folder, i_varia, N_varia, path)
        #for each link, store p_nom_opt of the link
        for i_link, link in n.links.iterrows():
            nomopt = link.p_nom_opt if link.p_nom_opt > smallM else 0
            optNoms[i_link] = optNoms[i_link] + [nomopt] if i_varia != 0 else [nomopt] 
    
            fullloadhours = sum(n.links_t.p0[i_link])/link.p_nom_opt if link.p_nom_opt >smallM else 0
            optFullLoad[i_link] = optFullLoad[i_link] + [fullloadhours] if i_varia != 0 else [fullloadhours] 
    
    
        for i_store, store in n.stores.iterrows():
            optNoms[i_store] = optNoms[i_store] + [store.e_nom_opt] if i_varia != 0 else [store.e_nom_opt] 
        for i_gen, gen in n.generators.iterrows():
            try:
                soldProduct = n.generators_t.p[i_gen]
                productPrice = n.generators.marginal_cost[i_gen] if n.generators.marginal_cost[i_gen] != 0 else n.generators_t.marginal_cost[i_gen]
                profit = sum(soldProduct*productPrice)
    
                soldAmounts[i_gen] = soldAmounts[i_gen] + [sum(soldProduct)] if i_varia != 0 else [sum(soldProduct)]
                profits[i_gen] = profits[i_gen] + [profit] if i_varia != 0 else [profit]
            except:
                soldAmounts[i_gen] = soldAmounts[i_gen] + [0] if i_varia != 0 else [0]
                profits[i_gen] = profits[i_gen] + [0] if i_varia != 0 else [0]
    
    #change datatype of results from list to array
    for key in optNoms.keys():
        optNoms[key] = np.array(optNoms[key],dtype=float)
    for key in soldAmounts.keys():
        soldAmounts[key] = -np.array(soldAmounts[key],dtype=float)
    for key in profits.keys():
        profits[key] = -np.array(profits[key],dtype=float)
    
    #change references to have all referenced on power demand, except CO2, CH4 and CH3OH (on main output)
    name = 'CHP_CC-S0000'
    optNoms[name] = optNoms[name] * n.links.efficiency[name] #change reference to CO2 output [t/h]
    name = 'CO2liquidification-S0000'
    optNoms[name] = optNoms[name] * (-n.links.efficiency2[name]) #change reference to power demand [MWh]
    name = 'H2compression-S0000'
    optNoms[name] = optNoms[name] * (-n.links.efficiency2[name]) #change reference to power demand [MWh]
    name = 'O2compression-S0000'
    optNoms[name] = optNoms[name] * (-n.links.efficiency2[name]) #change reference to power demand [MWh]
    name = 'CH3OHcatalysis-S0000'
    optNoms[name] = optNoms[name] * n.links.efficiency[name] #change reference to produced CH3OH [MWh]
    name = 'CH4catalysis-S0000'
    optNoms[name] = optNoms[name] * n.links.efficiency[name] #change reference to produced CH4 [MWh]
    return optNoms, optFullLoad, soldAmounts, profits, name_simulation, x_axis, x_label, x_unit

def plotSingleParamVar(name_folder, store_name, path):
    optNoms, optFullLoad, soldAmounts, profits, name_simulation, x_axis, x_label, x_unit = readResults(name_folder, path)
    plt.rcParams.update({'font.size': 8})
    fig = plt.figure(figsize=(7*2, 5/9.5*7))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 0.66, 0.66, 0.66], width_ratios = [1,1],hspace=0.1)

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[2], sharex=ax1)
    ax3 = plt.subplot(gs[4], sharex=ax1)
    ax4 = plt.subplot(gs[6], sharex=ax1)
    ax1_twin = ax1.twinx()
    ax2_twin = ax2.twinx()

    names = ['electrolyzer-S0000', 'H2compression-S0000', 'O2compression-S0000', 
             'CH3OHcatalysis-S0000', 'CH4catalysis-S0000', 'CO2liquidification-S0000','heatpump_60_120-S0000', 'heatrod-S0000',
             ]
    ax1.set_ylabel('Capacity [MW$_\mathrm{el}$]')
    y_prev = np.zeros_like(optNoms[names[0]])
    for name in names:
        y = y_prev + optNoms[name]
        y = fillEdges(y,y_prev)
        #we want to fill between y and y_prev when y > y_prev, however we also want a smooth transition between with and without a technology
        #so we first add a small offset to y for the first and the last point in which y = y_prev when there are different values in between
        if name in edgecolors.keys():
            ax1.fill_between(x_axis, y_prev, y, facecolor=colors[name],label=labels[name], where=(y > y_prev),edgecolor=edgecolors[name],hatch=hatches[name],linewidth=0.0)
        else:
            ax1.fill_between(x_axis, y_prev, y, facecolor=colors[name],label=labels[name], where=(y > y_prev),edgecolor=colors[name],linewidth=0.4)
        y_prev = y

        
    ax1_twin.set_ylabel('CC Capacity\n[t/h]')
    name = 'CHP_CC-S0000'
    ax1_twin.plot(x_axis, optNoms[name], color=colors[name], label=labels[name], markersize=markersizes[name],marker=markers[name])
    
    ax1.set_ylim(0,65)
    ax1.set_yticks([0,20,40,60])
    ax1_twin.set_ylim(0,21)
    ax1_twin.set_yticks([0,10,20])
    
    ax2.set_ylabel('Storage\nCapacity\n[MWh]')
    ax2_twin.set_ylabel('Storage\nPower\n[MW]')
    name = 'heatstore-S0000'
    ax2.plot(x_axis,optNoms[name],color=colors[name], label=labels[name], markersize=markersizes[name],marker=markers[name])
    name = 'heatcharge-S0000'
    ax2_twin.plot(x_axis,optNoms[name],color=colors[name], label=labels[name], markersize=markersizes[name],marker=markers[name])
    name = 'batstore-S0000'
    ax2.plot(x_axis,optNoms[name],color=colors[name], label=labels[name], markersize=markersizes[name],marker=markers[name])
    name = 'batcharge-S0000'
    ax2_twin.plot(x_axis,optNoms[name],color=colors[name], label=labels[name], markersize=markersizes[name],marker=markers[name])
    
    ax2_twin.set_ylim(0,65)
    ax2_twin.set_yticks([0,25,50])
    ax2.set_ylim(1,10000)
#     ax2.set_yticks([1,10,100,1000,10000])
    ax2.set_yscale('log')
    
    names = ['electrolyzer-S0000', 'H2compression-S0000', 'O2compression-S0000', 
             'CH3OHcatalysis-S0000', 'CH4catalysis-S0000', 'CO2liquidification-S0000','heatpump_60_120-S0000', 'heatrod-S0000',
             'CHP_CC-S0000',
             ]
    ax3.set_ylabel('Full Load\nHours [h]')
    for name in names:
        if not name in ['H2compression-S0000', 'O2compression-S0000', 'CH3OHcatalysis-S0000', 'CH4catalysis-S0000','electrolyzer-S0000']:
            ax3.plot(x_axis, optFullLoad[name], color=colors[name], label=labels[name], markersize=markersizes[name],
                 marker=markers[name])
        elif name == 'electrolyzer-S0000':
            ax3.plot(x_axis, optFullLoad[name], color=colors[name], label=labels[name], markersize=markersizes[name],
                 marker=markers[name],linewidth=3)
        elif name in ['H2compression-S0000', 'O2compression-S0000']:
            ax3.plot(x_axis, optFullLoad[name], color=colors[name], label=labels[name], markersize=markersizes[name],
                 marker=markers[name],markerfacecolor='k',markeredgecolor='k',linewidth=1.5)
        else:
            ax3.plot(x_axis, optFullLoad[name], color=colors[name], label=labels[name], markersize=markersizes[name],
                 marker=markers[name],markerfacecolor='none',markeredgecolor=colors[name],markeredgewidth=0.9)

    ax3.set_ylim(0,9500)
    ax3.set_yticks([0,8760/2,8760])
    
    
    ax4.set_ylabel('Revenue\nStreams\nRelative to\nRef. [%]')
    
    names = ['dagensell-S0000', 'H2_HP_market-S0000', 'CH3OH-S0000', 'CH4-S0000', 'CO2_L_market-S0000', 'O2_HP_market-S0000'
             ]
    y_prev = np.zeros_like(profits[names[0]])
    for name in names:
        y = y_prev + profits[name]/referenceProfits*100
        y = fillEdges(y,y_prev)
        #we want to fill between y and y_prev when y > y_prev, however we also want a smooth transition between with and without a technology
        #so we first add a small offset to y for the first and the last point in which y = y_prev when there are different values in between
        if name in edgecolors.keys():
            ax4.fill_between(x_axis, y_prev, y, facecolor=colors[name],label=labels[name], where=(y > y_prev),edgecolor=edgecolors[name],hatch=hatches[name],linewidth=0.0)
        else:
            ax4.fill_between(x_axis, y_prev, y, facecolor=colors[name],label=labels[name], where=(y > y_prev),edgecolor=colors[name])
        y_prev = y
    if name_folder == 'varyInvest_v2_20240201_180111':
        print(f'REFERENCE = {y*referenceProfits/100}')
        
    names = ['dagenbuy-S0000', 'CHP_exhaust-S0000'
             ]
    y_prev = np.zeros_like(profits[names[0]])
    for name in names:
        y = y_prev + profits[name]/referenceProfits*100
        y = -fillEdges(-y,-y_prev)
        #we want to fill between y and y_prev when y > y_prev, however we also want a smooth transition between with and without a technology
        #so we first add a small offset to y for the first and the last point in which y = y_prev when there are different values in between
        if name in edgecolors.keys():
            ax4.fill_between(x_axis, y_prev, y, facecolor=colors[name],label=labels[name], where=(y < y_prev),edgecolor=edgecolors[name],hatch=hatches[name],linewidth=0.0)
        else:
            ax4.fill_between(x_axis, y_prev, y, facecolor=colors[name],label=labels[name], where=(y < y_prev),edgecolor=colors[name])
        y_prev = y
        
        
    ax4.axhline(y=0, color='k', linewidth=ax4.spines['bottom'].get_linewidth())
    ax4.set_yticks([-100,-50,0,50,100])
        
    ax4.set_ylim(-100,120)
    
    delta = (min(x_axis)+max(x_axis))/2*0
    ax4.set_xlim((min(x_axis)-delta,max(x_axis)+delta))
    ax4.set_xlabel(x_label+' '+x_unit)
    ax1.axvline(x=x_reference[name_folder.split('_')[0]], color = 'dimgrey', linestyle = '--', linewidth=ax4.spines['bottom'].get_linewidth())
    ax2.axvline(x=x_reference[name_folder.split('_')[0]], color = 'dimgrey', linestyle = '--', linewidth=ax4.spines['bottom'].get_linewidth())
    ax3.axvline(x=x_reference[name_folder.split('_')[0]], color = 'dimgrey', linestyle = '--', linewidth=ax4.spines['bottom'].get_linewidth())
    ax4.axvline(x=x_reference[name_folder.split('_')[0]], color = 'dimgrey', linestyle = '--', linewidth=ax4.spines['bottom'].get_linewidth())
    
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    

    ax1.set_xlim(x_lims[name_folder.split('_')[0]])
    
    lines, legendlabels = ax1.get_legend_handles_labels()
    lines2, legendlabels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines + lines2, legendlabels + legendlabels2, loc='upper left',bbox_to_anchor=(1.12, 1.1),ncol=3)
    
    lines, legendlabels = ax2.get_legend_handles_labels()
    lines2, legendlabels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines + lines2, legendlabels + legendlabels2, loc='upper left',bbox_to_anchor=(1.12, 1.1),ncol=2)


    ax3.legend(loc='upper left',bbox_to_anchor=(1.12, 1.25),ncol=3)
    ax4.legend(loc='upper left',bbox_to_anchor=(1.12, 1),ncol=3)
    # Anzeigen des Diagramms
    #ax1.grid(True)
    plt.tight_layout()

    plt.savefig(store_name + '.svg', format='svg')
    
def plotMultiParamVar(name_folder, store_name, path):
    optNoms, optFullLoad, soldAmounts, profits, name_simulation, x_axis, x_label, x_unit = readResults(name_folder, path)
    plt.rcParams.update({'font.size': 8})
    fig, ((ax1,ax5),(ax2,ax6),(ax3,ax7),(ax4,ax8)) = plt.subplots(4,2,figsize=(7, 5/9.5*7), sharex='col')
    #gs = gridspec.GridSpec(4, 2, height_ratios=[1, 0.66, 0.66, 0.66], width_ratios = [0.66,0.33],hspace=0.1)
    
    ##################################################################################################################################
    #Linke Diagramme
    ##################################################################################################################################
    
    ax1_twin = ax1.twinx()
    ax2_twin = ax2.twinx()
    
    optNomsN = {}
    for key, value in optNoms.items():
        new_list = np.concatenate((value[1:5], [value[0]], value[5:9]))
        optNomsN[key] = new_list
    
    optFullLoadN = {}
    for key, value in optFullLoad.items():
        new_list = np.concatenate((value[1:5], [value[0]], value[5:9]))
        optFullLoadN[key] = new_list
        
    profitsN = {}
    for key, value in profits.items():
        # Create a new list with the desired order
        new_list = np.concatenate((value[1:5], [value[0]], value[5:9]))
        # Update the dictionary with the new list
        profitsN[key] = new_list
        
    x_names_arr = np.array(x_names)
    x_namesN1 = np.concatenate((x_names_arr[1:5], [x_names_arr[0]], x_names_arr[5:9])).tolist()
    
    x_axis = x_axis[0:9]
    
    names = ['electrolyzer-S0000', 'H2compression-S0000', 'O2compression-S0000', 
             'CH3OHcatalysis-S0000', 'CH4catalysis-S0000', 'CO2liquidification-S0000','heatpump_60_120-S0000', 'heatrod-S0000',
             ]
    y_prev = np.zeros_like(optNomsN[names[0]])
    for name in names:
        y = np.array(optNomsN[name])
        width = 0.7
    
        if name in edgecolors.keys():
            p = ax1.bar(x_axis, y, width, label=labels[name], bottom=y_prev,color=colors[name], hatch=hatches[name],edgecolor=edgecolors[name],linewidth=0)
        else:
            p = ax1.bar(x_axis, y, width, label=labels[name], bottom=y_prev,color=colors[name],edgecolor=colors[name],linewidth=0)
        y_prev = y_prev+y
    
    ax1.set_ylabel('Capacity\n[MW$_\mathrm{el}$]')
    
    #ax1_twin.set_ylabel('Installed CC\nCapacity [t/h]')
    name = 'CHP_CC-S0000'
    ax1_twin.plot(x_axis, optNomsN[name], 'o', color=colors[name], label=labels[name], markersize=4)
    
    ax1.set_ylim(0,60)
    ax1.set_yticks([0,20,40,60])
    ax1_twin.set_ylim(0,21)
    ax1_twin.set_yticks([0,10,20])
    
    ax2.set_ylabel('Storage\nCapacity\n[MWh]')
    #ax2_twin.set_ylabel('Power [MW]')
    name = 'heatstore-S0000'
    ax2.plot(x_axis,optNomsN[name],color=colors[name], label=labels[name], markersize=markersizes[name],marker=markers[name])
    name = 'heatcharge-S0000'
    ax2_twin.plot(x_axis,optNomsN[name],color=colors[name], label=labels[name], markersize=markersizes[name],marker=markers[name])
    name = 'batstore-S0000'
    ax2.plot(x_axis,optNomsN[name],color=colors[name], label=labels[name], markersize=markersizes[name],marker=markers[name])
    name = 'batcharge-S0000'
    ax2_twin.plot(x_axis,optNomsN[name],color=colors[name], label=labels[name], markersize=markersizes[name],marker=markers[name])
    
    ax2_twin.set_ylim(0,65)
    ax2_twin.set_yticks([0,25,50])
    ax2.set_ylim(1,10000)
    #     ax2.set_yticks([1,10,100,1000,10000])
    ax2.set_yscale('log')
    
    names = ['electrolyzer-S0000', 'H2compression-S0000', 'O2compression-S0000', 
             'CH3OHcatalysis-S0000', 'CH4catalysis-S0000', 'CO2liquidification-S0000','heatpump_60_120-S0000', 'heatrod-S0000',
             'CHP_CC-S0000',
             ]
    ax3.set_ylabel('Full Load\nHours [h]')
    for name in names:
        if not name in ['H2compression-S0000', 'O2compression-S0000', 'CH3OHcatalysis-S0000', 'CH4catalysis-S0000','electrolyzer-S0000']:
            ax3.plot(x_axis, optFullLoadN[name], color=colors[name], label=labels[name], markersize=markersizes[name],
                 marker=markers[name])
        elif name == 'electrolyzer-S0000':
            ax3.plot(x_axis, optFullLoadN[name], color=colors[name], label=labels[name], markersize=markersizes[name],
                 marker=markers[name],linewidth=3)
        elif name in ['H2compression-S0000', 'O2compression-S0000']:
            ax3.plot(x_axis, optFullLoadN[name], color=colors[name], label=labels[name], markersize=markersizes[name],
                 marker=markers[name],markerfacecolor='k',markeredgecolor='k',linewidth=1.5)
        else:
            ax3.plot(x_axis, optFullLoadN[name], color=colors[name], label=labels[name], markersize=markersizes[name],
                 marker=markers[name],markerfacecolor='none',markeredgecolor=colors[name],markeredgewidth=0.9)
    
    ax3.set_ylim(0,9500)
    ax3.set_yticks([0,8760/2,8760])
    
    
    ax4.set_ylabel('Revenue\nStreams\nRelative to\nRef. [%]')
    
    names = ['dagensell-S0000', 'H2_HP_market-S0000', 'CH3OH-S0000', 'CH4-S0000', 'CO2_L_market-S0000', 'O2_HP_market-S0000'
             ]
    y_prev = np.zeros_like(profitsN[names[0]])
    for name in names:
        y = y_prev + profitsN[name]/referenceProfits*100
        y = fillEdges(y,y_prev)
        #we want to fill between y and y_prev when y > y_prev, however we also want a smooth transition between with and without a technology
        #so we first add a small offset to y for the first and the last point in which y = y_prev when there are different values in between
        if name in edgecolors.keys():
            ax4.fill_between(x_axis, y_prev, y, facecolor=colors[name],label=labels[name], where=(y > y_prev),edgecolor=edgecolors[name],hatch=hatches[name],linewidth=0)
        else:
            ax4.fill_between(x_axis, y_prev, y, facecolor=colors[name],label=labels[name], where=(y > y_prev),edgecolor=colors[name],linewidth=0.4)
        y_prev = y
    
    names = ['dagenbuy-S0000', 'CHP_exhaust-S0000'
             ]
    y_prev = np.zeros_like(profitsN[names[0]])
    for name in names:
        y = y_prev + profitsN[name]/referenceProfits*100
        y = -fillEdges(-y,-y_prev)
        #we want to fill between y and y_prev when y > y_prev, however we also want a smooth transition between with and without a technology
        #so we first add a small offset to y for the first and the last point in which y = y_prev when there are different values in between
        if name in edgecolors.keys():
            ax4.fill_between(x_axis, y_prev, y, facecolor=colors[name],label=labels[name], where=(y <= y_prev),edgecolor=edgecolors[name],hatch=hatches[name],linewidth=0.0)
        else:
            ax4.fill_between(x_axis, y_prev, y, facecolor=colors[name],label=labels[name], where=(y <= y_prev),edgecolor=colors[name])
        y_prev = y
    
    ax4.axhline(y=0, color='k', linewidth=ax4.spines['bottom'].get_linewidth())
    ax4.set_yticks([-50,0,50,100,150])
    ax4.set_ylim(-80,165)
    
    
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    
    ax1.set_xlim(-1+0.5,9-0.5)
    ax4.set_xticks(x_axis)
    ax4.set_xticklabels(x_namesN1, rotation = 90)
    
    
    ##################################################################################################################################
    #Rechte Diagramme
    ##################################################################################################################################
    
    ax5_twin = ax5.twinx()
    ax6_twin = ax6.twinx()
    
    optNomsN = {}
    for key, value in optNoms.items():
        new_list = np.concatenate((value[9:13], [value[0]], value[13:17]))
        optNomsN[key] = new_list
    
    optFullLoadN = {}
    for key, value in optFullLoad.items():
        new_list = np.concatenate((value[9:13], [value[0]], value[13:17]))
        optFullLoadN[key] = new_list
        
    profitsN = {}
    for key, value in profits.items():
        new_list = np.concatenate((value[9:13], [value[0]], value[13:17]))
        profitsN[key] = new_list
        
    x_names_arr = np.array(x_names)
    x_namesN = np.concatenate((x_names_arr[9:13], [x_names_arr[0]], x_names_arr[13:17])).tolist()
    
    x_axis = x_axis[0:9]
    
    names = ['electrolyzer-S0000', 'H2compression-S0000', 'O2compression-S0000', 
             'CH3OHcatalysis-S0000', 'CH4catalysis-S0000', 'CO2liquidification-S0000','heatpump_60_120-S0000', 'heatrod-S0000',
             ]
    y_prev = np.zeros_like(optNomsN[names[0]])
    for name in names:
        y = np.array(optNomsN[name])
        width = 0.7
    
        if name in edgecolors.keys():
            p = ax5.bar(x_axis, y, width, label=labels[name], bottom=y_prev,color=colors[name], hatch=hatches[name],edgecolor=edgecolors[name],linewidth=0)
        else:
            p = ax5.bar(x_axis, y, width, label=labels[name], bottom=y_prev,color=colors[name],edgecolor=colors[name],linewidth=0)
        y_prev = y_prev+y
    
    #ax5.set_ylabel('Installed \nCapacity [MW]')
    
    ax5_twin.set_ylabel('CC\nCapacity\n[t/h]')
    name = 'CHP_CC-S0000'
    ax5_twin.plot(x_axis, optNomsN[name], 'o', color=colors[name], label=labels[name], markersize=4)
    
    ax5.set_ylim(0,60)
    ax5.set_yticks([0,20,40,60])
    ax5_twin.set_ylim(0,21)
    ax5_twin.set_yticks([0,10,20])
    
    #ax6.set_ylabel('Storage\nCapacity [MWh]')
    ax6_twin.set_ylabel('Storage\nPower\n[MW]')
    name = 'heatstore-S0000'
    ax6.plot(x_axis,optNomsN[name],color=colors[name], label=labels[name], markersize=markersizes[name],marker=markers[name])
    name = 'heatcharge-S0000'
    ax6_twin.plot(x_axis,optNomsN[name],color=colors[name], label=labels[name], markersize=markersizes[name],marker=markers[name])
    name = 'batstore-S0000'
    ax6.plot(x_axis,optNomsN[name],color=colors[name], label=labels[name], markersize=markersizes[name],marker=markers[name])
    name = 'batcharge-S0000'
    ax6_twin.plot(x_axis,optNomsN[name],color=colors[name], label=labels[name], markersize=markersizes[name],marker=markers[name])
    
    ax6_twin.set_ylim(0,65)
    ax6.set_ylim(1,10000)
    ax6_twin.set_yticks([0,25,50])
    ax6.set_yscale('log')
    
    names = ['electrolyzer-S0000', 'H2compression-S0000', 'O2compression-S0000', 
             'CH3OHcatalysis-S0000', 'CH4catalysis-S0000', 'CO2liquidification-S0000','heatpump_60_120-S0000', 'heatrod-S0000',
             'CHP_CC-S0000',
             ]
    #ax7.set_ylabel('Full Load\nHours [h]')
    for name in names:
        if not name in ['H2compression-S0000', 'O2compression-S0000', 'CH3OHcatalysis-S0000', 'CH4catalysis-S0000','electrolyzer-S0000']:
            ax7.plot(x_axis, optFullLoadN[name], color=colors[name], label=labels[name], markersize=markersizes[name],
                 marker=markers[name])
        elif name == 'electrolyzer-S0000':
            ax7.plot(x_axis, optFullLoadN[name], color=colors[name], label=labels[name], markersize=markersizes[name],
                 marker=markers[name],linewidth=3)
        elif name in ['H2compression-S0000', 'O2compression-S0000']:
            ax7.plot(x_axis, optFullLoadN[name], color=colors[name], label=labels[name], markersize=markersizes[name],
                 marker=markers[name],markerfacecolor='k',markeredgecolor='k',linewidth=1.5)
        else:
            ax7.plot(x_axis, optFullLoadN[name], color=colors[name], label=labels[name], markersize=markersizes[name],
                 marker=markers[name],markerfacecolor='none',markeredgecolor=colors[name],markeredgewidth=0.9)
    
    ax7.set_ylim(0,9500)
    ax7.set_yticks([0,8760/2,8760])
    
    
    #ax8.set_ylabel('Revenue Streams\nRelative to\nReference [%]')
    
    names = ['dagensell-S0000', 'H2_HP_market-S0000', 'CH3OH-S0000', 'CH4-S0000', 'CO2_L_market-S0000', 'O2_HP_market-S0000'
             ]
    y_prev = np.zeros_like(profitsN[names[0]])
    for name in names:
        y = y_prev + profitsN[name]/referenceProfits*100
        y = fillEdges(y,y_prev)
        #we want to fill between y and y_prev when y > y_prev, however we also want a smooth transition between with and without a technology
        #so we first add a small offset to y for the first and the last point in which y = y_prev when there are different values in between
        if name in edgecolors.keys():
            ax8.fill_between(x_axis, y_prev, y, facecolor=colors[name],label=labels[name], where=(y > y_prev),edgecolor=edgecolors[name],hatch=hatches[name],linewidth=0)
        else:
            ax8.fill_between(x_axis, y_prev, y, facecolor=colors[name],label=labels[name], where=(y > y_prev),edgecolor=colors[name],linewidth=0.4)
        y_prev = y
    
    names = ['dagenbuy-S0000', 'CHP_exhaust-S0000'
             ]
    y_prev = np.zeros_like(profitsN[names[0]])
    for name in names:
        y = y_prev + profitsN[name]/referenceProfits*100
        y = -fillEdges(-y,-y_prev)
        #we want to fill between y and y_prev when y > y_prev, however we also want a smooth transition between with and without a technology
        #so we first add a small offset to y for the first and the last point in which y = y_prev when there are different values in between
        if name in edgecolors.keys():
            ax8.fill_between(x_axis, y_prev, y, facecolor=colors[name],label=labels[name], where=(y <= y_prev),edgecolor=edgecolors[name],hatch=hatches[name],linewidth=0.0)
        else:
            ax8.fill_between(x_axis, y_prev, y, facecolor=colors[name],label=labels[name], where=(y <= y_prev),edgecolor=colors[name])
        y_prev = y
    
    ax8.axhline(y=0, color='k', linewidth=ax4.spines['bottom'].get_linewidth())
    ax8.set_yticks([-50,0,50,100,150])
    ax8.set_ylim(-80,165)
    
    
    plt.setp(ax1_twin.get_yticklabels(), visible=False)
    plt.setp(ax2_twin.get_yticklabels(), visible=False)
    plt.setp(ax5.get_yticklabels(), visible=False)
    plt.setp(ax6.get_yticklabels(), visible=False)
    plt.setp(ax7.get_yticklabels(), visible=False)
    plt.setp(ax8.get_yticklabels(), visible=False)
    
    ax5.set_xlim(-1+0.5,9-0.5)
    ax8.set_xticks(x_axis)
    ax8.set_xticklabels(x_namesN, rotation = 90)
    
    for axn in [ax1, ax5, ax2,ax3,ax6,ax7]:
        axn.axvline(x=4, color = 'dimgrey', linestyle = '--', linewidth=ax4.spines['bottom'].get_linewidth(), zorder=0)
    for axn in [ax4,ax8]:
        axn.axvline(x=4, color = 'dimgrey', linestyle = '--', linewidth=ax4.spines['bottom'].get_linewidth())
        
    
    # Anzeigen des Diagramms
    #ax1.grid(True)
    plt.tight_layout()

    plt.savefig(store_name + '.svg', format='svg')
    
# === plotting results of single optimization ===
    
    
def plotCarpetDAPrice(name_folder, i_varia, n_varia, store_name, path):
    '''
    carpet plot of DA price
    '''
    n = getVariationOutput(name_folder, i_varia, n_varia, path)
    hours = np.arange(1,25,1)
    days = np.arange(1,366,1)
    dagen = n.generators_t.marginal_cost['dagensell-S0000']
    data = np.resize(np.array(dagen), (len(days),len(hours)))
    data = data.transpose()
    plt.figure(figsize=(9,3))
    stretch = 5
    plt.imshow(data, cmap='viridis',extent=[0, data.shape[1], 0, data.shape[0]*stretch])
    plt.colorbar(label='Day-Ahead\nPrice [€/MWh]',shrink=1)  # Add a colorbar with a label
    
    plt.title('Day-Ahead Price [€/MWh]')
    plt.xlabel('Day [d]')
    plt.yticks(list(np.array([0,6,12,18,24])*stretch),labels=[0,6,12,18,24])
    plt.ylabel('Hour [h]')
    plt.savefig(store_name + '.svg', format='svg')
        
def plotCarpetDHLoad(name_folder, i_varia, n_varia, store_name, path):
    '''
    carpet plot of district heating load.
    '''
    n = getVariationOutput(name_folder, i_varia, n_varia, path)
    hours = np.arange(1,25,1)
    days = np.arange(1,366,1)
    data = np.resize(np.array(Q_fern), (len(days),len(hours)))
    plt.figure(figsize=(9,3))
    stretch = 5
    plt.imshow(data, cmap='viridis',extent=[0, data.shape[1], 0, data.shape[0]*stretch])
    plt.colorbar(label='Day-Ahead\nPrice [€/MWh]',shrink=1)  # Add a colorbar with a label
    
    plt.title('Day-Ahead Price [€/MWh]')
    plt.xlabel('Day [d]')
    plt.yticks(list(np.array([0,6,12,18,24])*stretch),labels=[0,6,12,18,24])
    plt.ylabel('Hour [h]')
    plt.savefig(store_name + '.svg', format='svg')  
    
def plotCarpetTESS_SoC(name_folder, i_varia, n_varia, store_name, path):
    '''
    carpet plot of thermal energy storage system State of Charge.
    '''
    n = getVariationOutput(name_folder, i_varia, n_varia, path)
    hours = np.arange(1,25,1)
    days = np.arange(1,366,1)
    data = n.stores_t.e['heatstore-S0000']/n.stores.e_nom_opt['heatstore-S0000']
    data = np.resize(np.array(data), (len(days),len(hours)))
    plt.figure(figsize=(9,3))
    stretch = 5
    plt.imshow(data, cmap='viridis',extent=[0, data.shape[1], 0, data.shape[0]*stretch])
    plt.colorbar(label='TESS SoC [%]',shrink=1)  # Add a colorbar with a label
    
    plt.title('TESS SoC [%]')
    plt.xlabel('Day [d]')
    plt.yticks(list(np.array([0,6,12,18,24])*stretch),labels=[0,6,12,18,24])
    plt.ylabel('Hour [h]')
    plt.savefig(store_name + '.svg', format='svg')
    
def plotFreqAnalysis(name_folder, i_varia, n_varia, store_name, path):
    '''
    FFT of TESS SoC, DH Load, DA Price
    '''
    n = getVariationOutput(name_folder, i_varia, n_varia, path)
    plt.rcParams.update({'font.size': 7})
    # Plot
    fig, ax = plt.subplots(figsize=(8.5/2.56, 5/2.54))
    
    plt.grid(which='both', zorder = 0, linestyle=':', linewidth=0.5)
    minor_locator = MultipleLocator(24)
    ax.xaxis.set_minor_locator(minor_locator)
    # Zeitintervall zwischen den Werten
    time_interval = 3600
    
    
    # === DH Demand ===
    data_values = np.array(Q_fern)
    # Berechnen Sie die FFT (Fast Fourier Transform) der geglätteten Wasserwerte
    fft_values = fft(data_values)
    # Frequenzen, die den FFT-Werten entsprechen
    frequencies = np.fft.fftfreq(len(df), d=time_interval)
    frequencies[0]=1e-9
    frequencies = 1 / frequencies
    frequencies = frequencies / 3600 #period in hours
    # Betrag der FFT-Werte
    fft_magnitude = np.abs(fft_values) / 40*1.4
    
    raster_cut = 800
    ax.plot(frequencies[raster_cut:], fft_magnitude[raster_cut:], color = 'r', marker = markers['batcharge-S0000'], label = 'DH Load', zorder = 2, markersize = 3, linewidth = 0.8,rasterized = True)
    ax.plot(frequencies[:raster_cut+1], fft_magnitude[:raster_cut+1], color = 'r', marker = markers['batcharge-S0000'],zorder = 2, markersize = 3, linewidth = 0.8)
    
    
    # === DA Price ===
    data_values = np.array(dagen)
    # Berechnen Sie die FFT (Fast Fourier Transform) der geglätteten Wasserwerte
    fft_values = fft(data_values)
    # Frequenzen, die den FFT-Werten entsprechen
    frequencies = np.fft.fftfreq(len(df), d=time_interval)
    frequencies[0]=1e-9
    frequencies = 1 / frequencies
    frequencies = frequencies / 3600 #period in hours
    # Betrag der FFT-Werte
    fft_magnitude = np.abs(fft_values) / 200000*1.9
    
    ax.plot(frequencies[raster_cut:], fft_magnitude[raster_cut:], color = colors['batcharge-S0000'], marker = markers['batcharge-S0000'], label = 'DA Price', zorder = 4, markersize = 3, linewidth = 0.8,rasterized = True)
    ax.plot(frequencies[:raster_cut+1], fft_magnitude[:raster_cut+1], color = colors['batcharge-S0000'], marker = markers['batcharge-S0000'], zorder = 4, markersize = 3, linewidth = 0.8)
    
    # === Heat storage SoC ===
    # Führen Sie die EMA-Berechnung separat durch
    data = n.stores_t.e['heatstore-S0000']/n.stores.e_nom_opt['heatstore-S0000']
    alpha = 1  # Passen Sie den Alpha-Wert an Ihre Bedürfnisse an
    data_smooth = data.ewm(alpha=alpha, adjust=False).mean()
    # Kopieren Sie die geglätteten Wasserwerte in ein NumPy-Array und stellen Sie sicher, dass sie ausgerichtet sind
    data_values = data_smooth.values
    # Berechnen Sie die FFT (Fast Fourier Transform) der geglätteten Wasserwerte
    fft_values = fft(data_values)
    # Frequenzen, die den FFT-Werten entsprechen
    frequencies = np.fft.fftfreq(len(df), d=time_interval)
    frequencies[0]=1e-9
    frequencies = 1 / frequencies
    frequencies = frequencies / 3600 #period in hours
    # Betrag der FFT-Werte
    fft_magnitude = np.abs(fft_values) / 800*1.2
    
    ax.plot(frequencies[raster_cut:], fft_magnitude[raster_cut:], color = colors['heatstore-S0000'], marker = markers['heatstore-S0000'], label = 'Heat Storage SoC', zorder = 3, markersize = 4, linewidth = 0.8,rasterized = True)
    ax.plot(frequencies[:raster_cut+1], fft_magnitude[:raster_cut+1], color = colors['heatstore-S0000'], marker = markers['heatstore-S0000'], zorder = 3, markersize = 4, linewidth = 0.8)
    ax.set_xlabel('Period [h]')
    ax.set_ylabel('Scaled Magnitude of\nFourier Transform [-]')
    ax.set_xlim(0, 548)
    ax.set_xticks(np.arange(0, 548, 7*24))
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.ylim(0,1.05)
    plt.legend(ncol=1)
    #plt.tight_layout()
    plt.show()
        
    plt.savefig(store_name + '.svg', format='svg')

def plotTimeseriesAnalysis(name_folder, i_varia, n_varia, store_name, path):
    '''
    FFT of TESS SoC, DH Load, DA Price
    '''
    n = getVariationOutput(name_folder, i_varia, n_varia, path)
    plt.rcParams.update({'font.size': 7})
    # Plot
    fig, ax = plt.subplots(figsize=(8.5/2.56, 5/2.54))
    ax_twin = ax.twinx()
    plt.grid(which='both',zorder=0,linestyle=':', linewidth=0.5)
    #minor_locator = MultipleLocator(24)
    #ax.xaxis.set_minor_locator(minor_locator)
    # Zeitintervall zwischen den Werten
    time_interval = 3600
    
    
    # === DH Demand ===
    data_values = np.array(Q_fern)/30*100
    x = np.arange(len(data_values))-8088#1560
    ax.plot(x, data_values, color = 'r', label = 'DH Load Point', zorder = 2, linewidth = 1)
    
    # === Heat storage SoC ===
    # Führen Sie die EMA-Berechnung separat durch
    data = n.links_t.p0['electrolyzer-S0000']/n.links.p_nom_opt['electrolyzer-S0000']*100
    data_values = data.values
    ax.plot(x, data_values, color = colors['electrolyzer-S0000'], label = 'Electrolyzer Load Point', zorder = 4, linewidth = 1.4)
    
    data = n.links_t.p0['CHP_CC-S0000']/n.links.p_nom_opt['CHP_CC-S0000']*100
    data_values = data.values
    ax.plot(x, data_values, color = 'k', label = 'CC Unit Load Point', zorder = 3, linewidth = 1.2, alpha=0.6)
    
    data = n.stores_t.e['heatstore-S0000']/n.stores.e_nom_opt['heatstore-S0000']*100
    data_values = data.values
    ax.plot(x, data_values, color = colors['heatstore-S0000'], label = 'Heat Storage SoC', zorder = 3, linewidth = 1.2)
    
    
    # === DA Price ===
    data_values = np.array(dagen)
    ax_twin.plot(x, data_values, color = colors['batcharge-S0000'], label = 'DA Price', zorder = 1, linewidth = 1, linestyle = '--')
    #ax_twin.plot([152,168,160,168,160], [22,22,28,22,16], color = colors['batcharge-S0000'], zorder = 0, linewidth = 0.6)
    
    
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('[-]')
    ax_twin.set_ylim(-0.05*250,250*1.05)
    ax.set_xlim(0,168)
    ax.set_ylim(-0.05*100,1.05*100)
    ax_twin.set_ylabel('DA Price [EUR/MWh]', color=colors['batcharge-S0000'])
    ax.set_ylabel('Load Point/SoC [%]')
    
    ax_twin.spines['right'].set_color(colors['batcharge-S0000'])  # Change color of the twin axis
    ax_twin.tick_params(axis='y', colors=colors['batcharge-S0000'])  # Change color of ticks
    
    # Change color of the labels
    for label in ax_twin.get_yticklabels():
        label.set_color(colors['batcharge-S0000'])  # Change color of tick labels
    #ax.set_xlim(0, 548)
    ax.set_xticks(np.arange(0, 7*24+1, 24))
    plt.tight_layout(rect=[0, 0, 1, 1])
    #plt.ylim(0,1.05)
    ax.legend(ncol=2,loc='upper center', bbox_to_anchor=(0.5, 1.3))
    #plt.tight_layout()
    plt.show()
    plt.savefig(store_name + '.svg', format='svg')