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
#%% === functions, should be in separate file in later version ===
# == functions for loading inputs ==
def getScenarios(preds,componentNames):
    """
    import boundary conditions of system (allows stochastic inputs)
    """

    keys = preds.keys()

    def getScenarioCombinations(keys):
        """
        Returns a Generator for the base Scenario Combinations
        """
        baseScenarios = []
        for key in keys:
            pred = preds[key][0]
            baseScenarios.append(pred.columns)
        return product(*baseScenarios)

    def getScenarioByCombination(combination:tuple):
        """
        Returns the Scenario, associated with the given combination. \\
        The order of the Tuple has to match the order of "keys"
        """
        scenario = {}
        p = 1
        for i,key in enumerate(keys):
            col = combination[i]
            pred = preds[key][0].loc[:,col]
            p_i = preds[key][1].loc[col]
            p *= p_i
            scenario[key] = pred
        return (scenario,p)

    combinations = getScenarioCombinations(keys)
    scenarios = {key:{} for key in keys}
    p_scenarios = {}
    for s,combination in enumerate(combinations):
            scenario,p = getScenarioByCombination(combination)
            p_scenarios[f"-S{s:04d}"] = p
            for key in keys:
                compName = componentNames[key]
                scenarios[key][f"{compName}-S{s:04d}"] = scenario[key]
    for key in keys:
        scenarios[key] = pd.DataFrame.from_dict(scenarios[key])
    return scenarios,p_scenarios

# == functions for parameter variations
def scaleUsingsinh(x,mean,inputscaling=300, cutoff = -500):
    '''
    used for scaling day ahead prices using sinh transform (for parameter variation)
    results in almost linear scaling up to inputscaling €/MWh, and more saturation-like behaviour for anything more extreme prices below cutoff are cut off and set to cutoff.
    '''
    y = np.sinh(x/inputscaling)
    fun = lambda z,a: np.mean(np.arcsinh(y*a)*inputscaling)
    popt, pcov = curve_fit(fun, [0],[mean],p0 = 1,bounds = (1e-6,100))
    a = popt[0]
    print(a)
    xscaled = np.arcsinh(y*a)*inputscaling
    return np.where(xscaled<=cutoff,cutoff,xscaled)

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

def writeVariationOutput(name_folder, i_varia, N_varia, network, path=None):
    '''
    write the output of a parameter variation to hdf5 format
    '''
    combined_path = os.path.join(path or '', name_folder)
    # Check if the directory exists, and create it if not
    if not os.path.exists(combined_path):
        os.makedirs(combined_path)
    # Export the network
    network.export_to_hdf5(os.path.join(combined_path, f'it_{i_varia}_{N_varia}.hdf5'))
    
# == functions for additional functionality in PyPSA ==
def extendComponentAttrs():
    '''
    Component attributes to deal with aFRR and multi-bus links
    '''
    attr = pypsa.descriptors.Dict({k: v.copy() for k, v in pypsa.components.component_attrs.items()})

    #Erweitert die Components comp und Storage Unit um eine Variable zur Vorhaltung der Reserveleistung
    comps = ("Generator","StorageUnit")

    for comp in comps:

        attr[comp].loc["r_pos_max"] = \
            ["static or series","MW",0.0,"Maximum reserve requirement","Input (optional)"]
        
        attr[comp].loc["r_neg_max"] = \
            ["static or series","MW",0.0,"Maximum reserve requirement","Input (optional)"]
        
        attr[comp].loc["r_pos"] = \
            ["series","MW",0.0,"Active reserve at bus","Output"]

        attr[comp].loc["r_neg"] = \
            ["series","MW",0.0,"Active reserve at bus","Output"]
        
    # attr["Markets"] = pd.DataFrame(
    #     columns=["type", "unit", "default", "description", "status"]
    # )
    # attr["Markets"].loc["name"] = \
    #     ["string",np.nan,np.nan,"Unique name","Input (required)"]
    
    attr["GlobalConstraint"].loc["bid_capacity"] = \
        ["series","MW(h)",0.0,"Capacity of resulting market bids","Output"]
    
    attr["GlobalConstraint"].loc["bid_price"] = \
        ["series","Euro/capacity",0.0,"Resulting market bids","Output"]
    
    #Das braucht man, um mehrere Outputs von einem Link zu erzeugen
    attr["Link"].loc["bus2"] = [
        "string",
        np.nan,
        np.nan,
        "2nd bus",
        "Input (optional)",
    ]
    attr["Link"].loc["efficiency2"] = [
        "static or series",
        "per unit",
        1.0,
        "2nd bus efficiency",
        "Input (optional)",
    ]
    attr["Link"].loc["p2"] = [
        "series",
        "MW",
        0.0,
        "2nd bus output",
        "Output",
    ]
    attr["Link"].loc["bus3"] = [
        "string",
        np.nan,
        np.nan,
        "3rd bus",
        "Input (optional)",
    ]
    attr["Link"].loc["efficiency3"] = [
        "static or series",
        "per unit",
        1.0,
        "3rd bus efficiency",
        "Input (optional)",
    ]
    attr["Link"].loc["p3"] = [
        "series",
        "MW",
        0.0,
        "3rd bus output",
        "Output",
    ]
    attr["Link"].loc["bus4"] = [
        "string",
        np.nan,
        np.nan,
        "4th bus",
        "Input (optional)",
    ]
    attr["Link"].loc["efficiency4"] = [
        "static or series",
        "per unit",
        1.0,
        "4th bus efficiency",
        "Input (optional)",
    ]
    attr["Link"].loc["p4"] = [
        "series",
        "MW",
        0.0,
        "4th bus output",
        "Output",
    ]
    attr["Link"].loc["bus5"] = [
        "string",
        np.nan,
        np.nan,
        "5th bus",
        "Input (optional)",
    ]
    attr["Link"].loc["efficiency5"] = [
        "static or series",
        "per unit",
        1.0,
        "5th bus efficiency",
        "Input (optional)",
    ]
    attr["Link"].loc["p5"] = [
        "series",
        "MW",
        0.0,
        "5th bus output",
        "Output",
    ]
    return attr

# = extra functionality in PyPSA for dealing with stochastic objective, aFRR, variable CHP efficiency =
def pos_reserve_constraints(network,snapshots):
    #Hier wird die positive SRL definiert

    #Hier werden die Bounds definiert also von 0 bis R_max, für SU und Gen 
    # Gl 4.5
    def gen_r_nom_bounds_pos(model, gen_name,snapshot):
        return (0,network.generators.at[gen_name,"r_pos_max"]) # Gebe die max pos Reserveleistung des Generators zurück

    def stor_r_nom_bounds_pos(model, stor_name, snapshot):
        return (0,network.storage_units.at[stor_name,"r_pos_max"]) # Gebe die max pos Reserveleistung der Storage unit zurück

    #Hier werden die Variablen gespeichert
    fixed_committable_gens_i = network.generators.index[~network.generators.p_nom_extendable & network.generators.committable]

    p_max_pu = get_switchable_as_dense(network, 'Generator', 'p_max_pu', snapshots)
    
    #Hier werden die Variablen initialisert
    network.model.generator_r_pos = Var(network.generators.index, snapshots, domain=Reals, bounds=gen_r_nom_bounds_pos)
    free_pyomo_initializers(network.model.generator_r_pos)

    #Gespeichert
    sus = network.storage_units
    fixed_sus_i = sus.index[~ sus.p_nom_extendable] # Only committable storage units allowed
    stor_p_max_pu = get_switchable_as_dense(network, 'StorageUnit', 'p_max_pu', snapshots)

    #Initialisiert
    network.model.storage_units_r_pos = Var(list(fixed_sus_i),snapshots,domain=Reals,bounds=stor_r_nom_bounds_pos)
    free_pyomo_initializers(network.model.storage_units_r_pos)

    #SU Gleichungen, auch zu finden in der BA 
    #Energie SU 
    # Gl 4.10 geändert
    stor_p_r_upper_soc_pos = {}
    for su in list(fixed_sus_i):
        # Umgang mit previous state of carge aus opf.py
        for i, sn in enumerate(snapshots):
            stor_p_r_upper_soc_pos[su, sn] = [[], ">=", 0.0]
            if i == 0:
                previous_state_of_charge = network.storage_units.at[su, "state_of_charge_initial"]
                stor_p_r_upper_soc_pos[su,sn][2] -= previous_state_of_charge
            else:
                previous_state_of_charge = network.model.state_of_charge[su, snapshots[i - 1]]
                stor_p_r_upper_soc_pos[su,sn][0].append((1,previous_state_of_charge))

            stor_p_r_upper_soc_pos[su, sn][0].extend([
                (1,network.model.storage_p_store[su,sn]),
                (-1,network.model.storage_p_dispatch[su,sn]),
                (-1.0,network.model.storage_units_r_pos[su,sn]), 
            ])
    l_constraint(network.model,"stor_p_r_upper_soc_pos",stor_p_r_upper_soc_pos, fixed_sus_i,snapshots)
    
    #leistung SU 
    # Gl 4.11 jedoch um store ergänzt
    stor_p_r_upper_dis_pos = {(stor,sn) :
                        [[(1,network.model.storage_p_dispatch[stor,sn]),                         #pDischarge
                          (-1,network.model.storage_p_store[stor,sn]),                           #TW Hat gefehlt
                        (1,network.model.storage_units_r_pos[stor,sn]),                                 #pRes
                        ],
                        "<=",stor_p_max_pu.at[sn,stor]*network.storage_units.p_nom[stor]]
                        for stor in list(fixed_sus_i) for sn in snapshots}

    l_constraint(network.model,"stor_p_r_upper_dis_pos",stor_p_r_upper_dis_pos, list(fixed_sus_i),snapshots)

    #Leistungsbegrenzung Gen
    # Gl. 4.12
    gen_p_r_upper_pos = {(gen,sn) :
                [[(1,network.model.generator_p[gen,sn]),
                (1,network.model.generator_r_pos[gen,sn]),
                (-p_max_pu.at[sn, gen]*network.generators.p_nom[gen],network.model.generator_status[gen,sn])
                ],
                "<=",0.]
                for gen in fixed_committable_gens_i for sn in snapshots}
    l_constraint(network.model, "generator_p_r_upper_pos", gen_p_r_upper_pos, list(fixed_committable_gens_i), snapshots)

def neg_reserve_constraints(network,snapshots):
    #Hier wird im Grunde genommen das gleiche wie bei der positiven Reserve gemacht, die Bounds sind auch hier von 0 bis R_max
    #Wichtig ist, dass nur Generatoren, die comittable sind, Reserve vorhalten können 
    def gen_r_nom_bounds_neg(model, gen_name,snapshot):
        return (0,network.generators.at[gen_name,"r_neg_max"])

    def stor_r_nom_bounds_neg(model, stor_name, snapshot):
        return (0,network.storage_units.at[stor_name,"r_neg_max"])

    fixed_committable_gens_i = network.generators.index[~network.generators.p_nom_extendable & network.generators.committable]


    p_min_pu = get_switchable_as_dense(network, 'Generator', 'p_min_pu', snapshots)

    network.model.generator_r_neg = Var(list(network.generators.index), snapshots, domain=Reals, bounds=gen_r_nom_bounds_neg)
    free_pyomo_initializers(network.model.generator_r_neg)

    
    sus = network.storage_units
    fixed_sus_i = sus.index[~ sus.p_nom_extendable] # Only committable storage units allowed
    stor_p_min_pu = get_switchable_as_dense(network, 'StorageUnit', 'p_min_pu', snapshots)

    network.model.storage_units_r_neg = Var(list(fixed_sus_i),snapshots,domain=Reals,bounds=stor_r_nom_bounds_neg)
    free_pyomo_initializers(network.model.storage_units_r_neg)

    # Gl. 4.15 (geändert)
    stor_p_r_lower_soc_neg = {}
    for su in list(fixed_sus_i):
        # Umgang mit previous state of carge aus opf.py
        for i, sn in enumerate(snapshots):
            stor_p_r_lower_soc_neg[su, sn] = [[], "<=", network.storage_units.p_nom[su]]
            if i == 0:
                previous_state_of_charge = network.storage_units.at[su, "state_of_charge_initial"]
                stor_p_r_lower_soc_neg[su,sn][2] -= previous_state_of_charge
            else:
                previous_state_of_charge = network.model.state_of_charge[su, snapshots[i - 1]]
                stor_p_r_lower_soc_neg[su,sn][0].append((1,previous_state_of_charge))

            stor_p_r_lower_soc_neg[su, sn][0].extend([
                (1,network.model.storage_p_store[su,sn]),
                (-1,network.model.storage_p_dispatch[su,sn]),
                (+1.0,network.model.storage_units_r_neg[su,sn]), 
            ])
    l_constraint(network.model,"stor_p_r_lower_soc_neg",stor_p_r_lower_soc_neg, fixed_sus_i,snapshots)

    # Gl 4.16, jedoch um dispatch ergänzt
    stor_p_r_upper_dis_neg = {(stor,sn) :
                        [[(1,network.model.storage_p_store[stor,sn]),                         #pDischarge
                          (-1,network.model.storage_p_dispatch[stor,sn]),                      #TW pCharge
                        (1,network.model.storage_units_r_neg[stor,sn]),                                 #pRes
                        ],
                        "<=",-1.0*stor_p_min_pu.at[sn,stor]*network.storage_units.p_nom[stor]]
                        for stor in list(fixed_sus_i) for sn in snapshots}

    l_constraint(network.model,"stor_p_r_upper_dis_neg",stor_p_r_upper_dis_neg, list(fixed_sus_i),snapshots)

    # statt Gl. 4.17 & 4.18
    gen_p_r_lower_neg = {(gen,sn) :
                [[(1,network.model.generator_p[gen,sn]),
                (-1,network.model.generator_r_neg[gen,sn]),
                (-p_min_pu.at[sn, gen]*network.generators.p_nom[gen],network.model.generator_status[gen,sn])
                ],
                ">=",0.]
                for gen in fixed_committable_gens_i for sn in snapshots}
    l_constraint(network.model, "generator_p_r_lower_neg", gen_p_r_lower_neg, list(fixed_committable_gens_i), snapshots)

def top_iso_fuel_line(model, snapshot,s):
    return ( model.link_p['CHP_elec'+s, snapshot]
#             +model.generator_r_pos['genloadgas'+s,snapshot]
            <= operatingChar_coefBoilerLoad * model.generator_p['CHP_fuelinput'+s,snapshot]
            + operatingChar_coefDHLoad * model.link_p['CHP_heat'+s, snapshot] 
            + operatingChar_coefConst)

def cor_res(model, snapshot,s):
    # Muss eingeführt werden, um die negative Reserve richtig darzustellen
    return (model.link_p["CHP_elec"+s, snapshot]>=model.generator_r_neg['CHP_fuelinput'+s,snapshot])

def four_hour_comittment_gen(model,snapshot4H,offset,s,sign):
    freq = snapshot4H.freq / 4
    if sign == "pos":
        r = model.generator_r_pos
    else:
        r = model.generator_r_neg    
    return r["genloadgas"+s,snapshot4H+freq*offset] == r["genloadgas"+s,snapshot4H+freq*(1+offset)]

def four_hour_comittment_bat(model,snapshot4H,offset,s,sign):
    freq = snapshot4H.freq / 4
    if sign == "pos":
        r = model.storage_units_r_pos
    else:
        r = model.storage_units_r_neg   
    return r["bat"+s,snapshot4H+freq*offset] == r["bat"+s,snapshot4H+freq*(1+offset)]

def stochasticOpt(network,snapshots):
    """
    Definiert den Zusammenhang zwischen DA-Gen und der Liste der Angebote (aus den verschiedenen Szenarien) neu
    Es werden noch keine Kosten aufgestellt. 
    Ähnlich für die Regelleistung: Summe der Angebote wird gebildet.
    """
    # Fixiere die Bids, wenn die Auktion beendet wurde
    def isFixed(bidName): 
        if not bidName in network.global_constraints_t.bid_capacity.columns:
            return False
        if network.global_constraints_t.bid_capacity.loc[snapshots,bidName].isna().any().any():
            return False
        return True
    
    if isFixed("Da"):
        columns = network.global_constraints_t.bid_price[["Da"]].columns
        network.priceLevels[columns] = network.global_constraints_t.bid_price[columns]
    if isFixed("aFRR_pos"): 
        columns = network.global_constraints_t.bid_price[["aFRR_pos"]].columns
        network.priceLevels[columns] = network.global_constraints_t.bid_price[columns]
    if isFixed("aFRR_neg"):
        columns = network.global_constraints_t.bid_price[["aFRR_neg"]].columns
        network.priceLevels[columns] = network.global_constraints_t.bid_price[columns]

    network.priceLevels = network.priceLevels[["Da","aFRR_pos","aFRR_neg"]] # Order columns of PriceLevels
    priceLevels = network.priceLevels

    #init Price index
    network.model.J = RangeSet(0,len(priceLevels["Da"].iloc[0])-1)
    J = network.model.J
    network.model.K = RangeSet(0,len(priceLevels["aFRR_pos"].iloc[0])-1)
    K = network.model.K
    network.model.L = RangeSet(0,len(priceLevels["aFRR_neg"].iloc[0])-1)
    L = network.model.L

    S = network.scenarios

    #Festlegen auf welchem Preisniveau ein Angebot abgegen wird
    #Init Variables
    network.model.Da_bid= Var(snapshots,S,J,domain=Reals,bounds=(-500,3000))
    free_pyomo_initializers(network.model.Da_bid)

    network.model.aFRR_pos_bid= Var(snapshots,S,K, domain=NonNegativeReals)
    free_pyomo_initializers(network.model.aFRR_pos_bid)

    network.model.aFRR_neg_bid= Var(snapshots,S,L, domain=NonNegativeReals)
    free_pyomo_initializers(network.model.aFRR_neg_bid)

    network.model.Da_dispatch= Var(snapshots,S,domain=Reals)
    free_pyomo_initializers(network.model.Da_dispatch)

    network.model.aFRR_pos_dispatch= Var(snapshots,S,K, domain=NonNegativeReals)
    free_pyomo_initializers(network.model.aFRR_pos_dispatch)

    network.model.aFRR_neg_dispatch= Var(snapshots,S,L, domain=NonNegativeReals)
    free_pyomo_initializers(network.model.aFRR_neg_dispatch)

    # fix bids
    if isFixed("Da"):
        for s in S:
            for j in J:
                for sn in snapshots:
                    network.model.Da_bid[sn,s,j].fix(network.global_constraints_t.bid_capacity.at[sn,("Da",j)])
    if isFixed("aFRR_pos"):
        for s in S:
            for k in K:
                for sn in snapshots:
                    network.model.aFRR_pos_bid[sn,s,k].fix(network.global_constraints_t.bid_capacity.at[sn,("aFRR_pos",k)])
    if isFixed("aFRR_neg"):
        for s in S:
            for l in L:
                for sn in snapshots:
                    network.model.aFRR_neg_bid[sn,s,l].fix(network.global_constraints_t.bid_capacity.at[sn,("aFRR_neg",l)])
                    
    # Bids must be equal for all scenarios
    Da_bids_equal = {}
    for sn in snapshots:
        for i, s in enumerate(S[:-1]):
            for j in J:
                    Da_bids_equal[sn,s,j] = [[], "==", 0.0]
                    Da_bids_equal[sn,s,j][0] =[
                        ( 1,network.model.Da_bid[sn,s,j]),
                        (-1,network.model.Da_bid[sn,S[i+1],j])
                    ]
    l_constraint(network.model,"Da_bids_equal",Da_bids_equal,snapshots,S[:-1],J)
    
    aFRR_pos_bids_equal = {}
    for sn in snapshots:
        for i, s in enumerate(S[:-1]):
            for k in K:
                    aFRR_pos_bids_equal[sn,s,k] = [[], "==", 0.0]
                    aFRR_pos_bids_equal[sn,s,k][0] =[
                        ( 1,network.model.aFRR_pos_bid[sn,s,k]),
                        (-1,network.model.aFRR_pos_bid[sn,S[i+1],k])
                    ]
    l_constraint(network.model,"aFRR_pos_bids_equal",aFRR_pos_bids_equal,snapshots,S[:-1],K)
    
    aFRR_neg_bids_equal = {}
    for sn in snapshots:
        for i, s in enumerate(S[:-1]):
            for l in L:
                    aFRR_neg_bids_equal[sn,s,l] = [[], "==", 0.0]
                    aFRR_neg_bids_equal[sn,s,l][0] =[
                        ( 1,network.model.aFRR_neg_bid[sn,s,l]),
                        (-1,network.model.aFRR_neg_bid[sn,S[i+1],l])
                    ]
    l_constraint(network.model,"aFRR_neg_bids_equal",aFRR_neg_bids_equal,snapshots,S[:-1],L)

    sus = network.storage_units.index[~ network.storage_units.p_nom_extendable] # Only committable storage units allowed
    gens = network.generators.index[~network.generators.p_nom_extendable & network.generators.committable]


    # Sum up all accepted bids (dispatch) per scenario and timestep and make them equal to the szenarios reserves
    aFRR_pos_dispatch_sum = {}
    for sn in snapshots:
        for s in S:
            aFRR_pos_dispatch_sum[sn,s] = [[], "==", 0.0]

            for k in K:
                aFRR_pos_dispatch_sum[sn,s][0].append((-1,network.model.aFRR_pos_dispatch[sn,s,k]))

            gens_s = gens[gens.str.endswith(s)]
            sus_s = sus[sus.str.endswith(s)]
            for gen in gens_s:
                aFRR_pos_dispatch_sum[sn,s][0].append((1,network.model.generator_r_pos[gen,sn]))
            for su in sus_s:
                aFRR_pos_dispatch_sum[sn,s][0].append((1,network.model.storage_units_r_pos[su,sn]))
    #Angebotene Regelleistung = Summe aus Generator+Storage                    
    l_constraint(network.model,"aFRR_pos_dispatch_sum",aFRR_pos_dispatch_sum,snapshots,S) 

    aFRR_neg_dispatch_sum = {}
    for sn in snapshots:
        for s in S:
            aFRR_neg_dispatch_sum[sn,s] = [[], "==", 0.0]

            for l in L:
                aFRR_neg_dispatch_sum[sn,s][0].append((-1,network.model.aFRR_neg_dispatch[sn,s,l]))

            gens_s = gens[gens.str.endswith(s)]
            sus_s = sus[sus.str.endswith(s)]
            for gen in gens_s:
                aFRR_neg_dispatch_sum[sn,s][0].append((1,network.model.generator_r_neg[gen,sn]))
            for su in sus_s:
                aFRR_neg_dispatch_sum[sn,s][0].append((1,network.model.storage_units_r_neg[su,sn]))
                        
    l_constraint(network.model,"aFRR_neg_dispatch_sum",aFRR_neg_dispatch_sum,snapshots,S)

    
    # Sum up all accepted da bids to dispatch
    gamma = {}
    for sn in snapshots:
        for s in S:
            for j in J:
                #JOHHERE changed .loc[j] to .iloc[j] otherwise error
                # Für jedes Szenario s iteriere über die verschiedene Preisniveaus j 
                # marginal cost von dagen generator = Preise von aktuellem Szenario
                gamma[sn,s,j] = 1 if priceLevels.loc[sn,"Da"].iloc[j] <= network.generators_t.marginal_cost.at[sn,"dagensell"+s] else 0 #Veränderung
                # Stelle das gamma für jedes Szenario auf, um die Preise des darunter liegenden Szenario mitzunehmen
      
    # Für jedes Szenario s und für jedes Preisniveau j: da_dispatch = dispatch_sum 
    # Gamma entscheidet, ob das Preisniveau dazugerechnet wird.
    Da_dispatch_sum = {}
    for sn in snapshots:
        for s in S:
            Da_dispatch_sum[sn,s] = [[], "==", 0.0]
            Da_dispatch_sum[sn,s][0].append((-1,network.model.Da_dispatch[sn,s]))
            for j in J:
                Da_dispatch_sum[sn,s][0].append((gamma[sn,s,j],network.model.Da_bid[sn,s,j]))
                        
    l_constraint(network.model,"Da_dispatch_sum",Da_dispatch_sum,snapshots,S)

# JOHHERE commented out threw error
#     Da_bid_unused ={}
#     for sn in snapshots:
#         for j in J:
#             if not any([gamma[sn,s,j] for s in S]): # Wenn alle gamma = 0 -> Wenn Szenariopreis unterhalb der aller Preisniveaus
#                 Da_bid_unused[sn,j] = [(1,network.model.Da_bid[sn,s,j])],"==",0
#             else:
#                 Da_bid_unused[sn,j] = [],"==",0

#     l_constraint(network.model,"Da_bid_unused",Da_bid_unused,snapshots,J)


    beta = {}
    for sn in snapshots:
        for s in S:
            for k in K:
                #JOHHERE again iloc instead of loc
                beta[sn,s,k] = 1 if priceLevels.loc[sn,"aFRR_pos"].iloc[k] <= network.aFRR_pos.loc[sn,"aFRRpos"+s] else 0

    aFRR_pos_dispatch_accepted ={}
    for sn in snapshots:
        for s in S:
            for k in K:
                aFRR_pos_dispatch_accepted[sn,s,k] = [[],"==",0]
                aFRR_pos_dispatch_accepted[sn,s,k][0] = [
                    (1,network.model.aFRR_pos_dispatch[sn,s,k]),
                    (-beta[sn,s,k],network.model.aFRR_pos_bid[sn,s,k])
                ]

    l_constraint(network.model,"aFRR_pos_dispatch_accepted",aFRR_pos_dispatch_accepted,snapshots,S,K)

    beta = {}
    for sn in snapshots:
        for s in S:
            for l in L:
                #JOHHERE again iloc instead of loc
                beta[sn,s,l] = 1 if priceLevels.loc[sn,"aFRR_neg"].iloc[l] <= network.aFRR_neg.loc[sn,"aFRRneg"+s] else 0

    aFRR_neg_dispatch_accepted ={}
    for sn in snapshots:
        for s in S:
            for l in L:
                aFRR_neg_dispatch_accepted[sn,s,l] = [[],"==",0]
                aFRR_neg_dispatch_accepted[sn,s,l][0] = [
                    (1,network.model.aFRR_neg_dispatch[sn,s,l]),
                    (-beta[sn,s,l],network.model.aFRR_neg_bid[sn,s,l])
                ]

    l_constraint(network.model,"aFRR_neg_dispatch_accepted",aFRR_neg_dispatch_accepted,snapshots,S,L)

    # Map Da dispatch to Da Generator -> Summe von da dispatch und durch generator bestellter Leistung = 0 
    # Bedingung: Dispatch-Tabelle definiert wie viel der Generator erzeugen muss
    Da_dispatch_gen = {}
    for sn in snapshots:
        for s in S:
            Da_dispatch_gen[sn,s] = [[], "==", 0.0]
            Da_dispatch_gen[sn,s][0] =[
                ( 1,network.model.Da_dispatch[sn,s]),
                ( 1,network.model.generator_p["dagensell"+s,sn]),
                ( 1,network.model.generator_p["dagenbuy"+s,sn])    #Änderung
            ]
    l_constraint(network.model,"Da_dispatch_gen",Da_dispatch_gen,snapshots,S)

def redefine_linear_objective(network, snapshots):
    """
    v1 von Alois, dann abgeändert damit es mit Toms code zusammenarbeiten kann
    Geht nur im Fall von einem einzigen Szenario (da keine Wahrscheinlichkeits-Gewichtung)
    """
    #Hier wird die Objective Funktion (also Kostenfunktion) um die Therme der SRL erweitert.
    #Auch hier müssen die Terme immer manuell hinzugefügt werden, was noch bearbeitet gehört.
    #neg und pos Reserve sind gleich aufgebaut, nur die Preise sind natürlich verschieden

    oldObjCoefs = network.model.objective.expr.linear_coefs #Kostenfunktion hier enthalten in network.model.objective.expr
    oldObjVars = network.model.objective.expr.linear_vars 

    S = network.scenarios
    for sn in snapshots:    # Erweitere die Kostenfunktion um die Regelleistung
        for s in S:
            for k in network.model.K:
                oldObjCoefs.append(network.priceLevels.loc[sn,"aFRR_pos"].iloc[k])
                oldObjVars.append(network.model.aFRR_pos_dispatch[sn,s,k])
            
            for l in network.model.L:
                oldObjCoefs.append(network.priceLevels.loc[sn,"aFRR_neg"].iloc[l])
                oldObjVars.append(network.model.aFRR_neg_dispatch[sn,s,l])

    oldObjConst = network.model.objective.expr.constant
    oldObjSense = network.model.objective.sense

    
    index = range(len(oldObjCoefs))
    network.model.del_component(network.model.objective) # Lösche die alte Kostenfunktion
    #Erweitere die alte Kostenfunktion um die neuen Objectives und speicher sich wieder ab
    network.model.objective = Objective(expr=sum(oldObjVars[i]*oldObjCoefs[i] for i in index)+oldObjConst, sense=oldObjSense)
    # Das print könnte entfernt werden, zeigt aber immer ganz gut wie die Kostenfunktion aufgebaut wird
    # print(network.model.objective.expr.to_string())

# def stochasticObjective(network,snapshots):
#     """
#     Summenfunktion, die die Kosten und Gewinne aufaddiert. 
#     Dafür wird jedes Szenario mit der Wahrscheinlichkeit p multipliziert und aufaddiert
#     """
#     S = network.scenarios
#     cost =-sum([
#         network.p_scenarios[s]*sum([
#             -network.generators.at["genloadgas-S0000","marginal_cost"]*network.model.generator_p["genloadgas"+s,sn]
#             +network.generators_t.marginal_cost.at[sn,"dagen"+s]*network.model.Da_dispatch[sn,s] # funkt. nicht mit neuem DA-Generator
#             +sum([
#               network.priceLevels.loc[sn,"aFRR_pos"].iloc[k]*network.model.aFRR_pos_dispatch[sn,s,k]
#             for k in network.model.K])
#             +sum([
#               network.priceLevels.loc[sn,"aFRR_neg"].iloc[l]*network.model.aFRR_neg_dispatch[sn,s,l]
#             for l in network.model.L])
#             -network.generators.at["penalty-S0000","marginal_cost"]*network.model.generator_p["penalty"+s,sn]
#         for sn in snapshots]) 
#     for s in S])
#     network.model.del_component(network.model.objective)
#     objective = Objective(expr = cost,sense=minimize)
#     objective.construct()
#     # print(objective.expr.to_string().replace("+","\n +"))
#     network.model.objective = objective

def extra_functionality(n, snapshots):
#     # Rangeset als Iterator für Scenarios

    neg_reserve_constraints(n,snapshots)
    pos_reserve_constraints(n,snapshots)

    n.model.top_iso_fuel_line = Constraint(snapshots,n.scenarios, rule = top_iso_fuel_line)
#     n.model.cor_res = Constraint(snapshots,n.scenarios, rule = cor_res)
    assert len(snapshots)%4==0
#     n.model.four_hour_comittment_gen = Constraint(snapshots[::4],range(3),n.scenarios,("pos","neg"),rule = four_hour_comittment_gen)
#     #n.model.four_hour_comittment_bat = Constraint(snapshots[::4],range(3),n.scenarios,("pos","neg"),rule = four_hour_comittment_bat)
    stochasticOpt(n,snapshots)
#     #stochasticObjective(n,snapshots) #ausgeklammert weil es sonst für jedes neues Teil der Objective explizit erweitert werden soll
    redefine_linear_objective(n, snapshots)

# # # iis code from https://groups.google.com/g/pypsa/c/UDFnQAyILWg
# # solver_parameters = "ResultFile=model.ilp" # write an ILP file to print the IIS
# # n.model = None
# # n.model = pypsa.opf.network_lopf_build_model(n,n.snapshots,formulation="kirchhoff")
# # extra_functionality(n,n.snapshots)
# # opt = pypsa.opf.network_lopf_prepare_solver(n, solver_name="gurobi")
# # n.results=opt.solve(n.model, options_string=solver_parameters,tee=True)
# # n.results.write()

def extra_postprocessing(network, snapshots, duals):
    """
    Extracts the new results and adds them to the pypsa network.
    Results are written to:
    network.generators_t.r_pos
    network.generators_t.r_neg
    network.storage_units_t.r_pos
    network.storage_units_t.r_neg
    network.global_constraints_t.bid_capacity
    network.global_constraints_t.bid_price
    """
    allocate_series_dataframes(
        network,
        {
            "Generator": ["r_pos","r_neg"],
            "StorageUnit": ["r_pos","r_neg"],
        },
    )
    if not len(network.global_constraints_t.bid_capacity):
        allocate_series_dataframes(network,{"GlobalConstraint": ["bid_capacity","bid_price"]})

    # from opf.py
    def clear_indexedvar(indexedvar):
        for v in indexedvar._data.values():
            v.clear()

    def get_values(indexedvar, free=False):
        s = pd.Series(indexedvar.get_values(), dtype=float)
        if free:
            clear_indexedvar(indexedvar)
        return s

    def set_from_series(df, series):
        df.loc[snapshots] = series.unstack(0).reindex(columns=df.columns)

    def set_from_series(df, series):
        df.loc[snapshots] = series.unstack(0).reindex(columns=df.columns)

    model = network.model

    if len(network.generators):
        set_from_series(network.generators_t.r_pos, get_values(model.generator_r_pos))
        set_from_series(network.generators_t.r_neg, get_values(model.generator_r_neg))

    if len(network.storage_units):
        set_from_series(network.storage_units_t.r_pos, get_values(model.storage_units_r_pos))
        set_from_series(network.storage_units_t.r_neg, get_values(model.storage_units_r_neg))

    prices  = [model.Da_bid,model.aFRR_pos_bid,model.aFRR_neg_bid]
    bid_price = network.priceLevels.copy()
    columns = bid_price.columns

    bid_capacity = pd.concat([get_values(price).xs("-S0000",level=1).unstack() for price in prices],axis=1)
    bid_capacity.columns = columns

    network.global_constraints_t.bid_capacity = network.global_constraints_t.bid_capacity.reindex(columns=columns)
    network.global_constraints_t.bid_price = network.global_constraints_t.bid_price.reindex(columns=columns)
    network.global_constraints_t.bid_capacity[columns] = bid_capacity.loc[snapshots]
    network.global_constraints_t.bid_price[columns] = bid_price.loc[snapshots]

#%% === prepare inputs ===
# == choose inputs to use ==
snapshots = pd.date_range("2023-01-01","2023-12-31 23:00",freq="H") 
df = ladeVorverarbeiteteDaten(name="2023",sampleInterval="1H", filled=False).loc[snapshots]  
inputFile = r'InputData\Parameter_Input_2023.csv'
#inputFile = r'InputData\Parameter_Input_2050.csv'

# == assign prices to variables, note: some of these are hard-coded! ==
# aFRR prices are loaded, but not used in this example, as no network element can offer aFRR
aFRR_pos = df[["pos_aFRR_[EURO/MW]"]]
aFRR_pos.columns = [0]
aFRR_neg = df[["neg_aFRR_[EURO/MW]"]]
aFRR_neg.columns = [0]
Da = df[["Da_[EUR/MWh]"]]
Da.columns = [0]
Q_fern = df[["Q_FernW_[MW]"]]
Q_fern.columns = [0]
p_1 = pd.Series({0:1}) 
preds = {}
preds["aFRR_pos"] = (aFRR_pos,p_1)
preds["aFRR_neg"] = (aFRR_neg,p_1)
preds["Da"] = (Da,p_1)
preds["Q_fern"] = (Q_fern,p_1)

scenarios,p_scenarios = getScenarios(preds,{"aFRR_pos":"aFRRpos","aFRR_neg":"aFRRneg","Da":"dagensell","Q_fern":"loadheat"})

# == calculate the variable efficiency of the extraction-condensing CHP ===
operatingChar_coefBoilerLoad = 1/2.71904565
operatingChar_coefDHLoad = -0.61319625/2.71904565
operatingChar_coefConst = -24.73506702/2.71904565
# alternative: calculate based on operating points (change as necessary)
#operatingPoints = np.array([[59.74,13.91,0],[59.74,9.204,30],[46.01,6.399,22.01],[46.01,10.22,0]])
#def operatingCharacteristicPgen(X,a,b,c):
#    '''maximal Pgen under given X = firing load, DH load'''
#    firingLoad = X[0]
#    DHLoad = X[1]
#    return a*firingLoad+b*DHLoad+c
##now fit operatingPoint data on characteristic to find a,b,c:
#X = (operatingPoints[:,0],operatingPoints[:,2])
#Y = operatingPoints[:,1]
#popt,pcov = curve_fit(operatingCharacteristicPgen,X,Y)
##following params go into boundary condition
#operatingChar_coefBoilerLoad = popt[0]
#operatingChar_coefDHLoad = popt[1]
#operatingChar_coefConst = popt[2]

# == load techno-economical parameters ==
# = change correction for capex depending on amortisation method =
capexCorrection = len(snapshots)/24 / 365.25  #linear amortization, can be improved by considering annuity w discount rate!

# = load input data =
dfins = pd.DataFrame()
dfins = pd.read_csv(inputFile, index_col=0)
# = compute investment cost to consider in the optimization based on total cost, lifetime and capexCorrection =
dfins.at['heatrod','capcost'] = dfins.at['heatrod','capcost_total']/dfins.at['heatrod','lifetime']*capexCorrection
dfins.at['heatpump_25_120','capcost'] = dfins.at['heatpump_25_120','capcost_total']/dfins.at['heatpump_25_120','lifetime']*capexCorrection
dfins.at['heatpump_25_60','capcost'] = dfins.at['heatpump_25_60','capcost_total']/dfins.at['heatpump_25_60','lifetime']*capexCorrection
dfins.at['heatpump_60_120','capcost'] = dfins.at['heatpump_60_120','capcost_total']/dfins.at['heatpump_60_120','lifetime']*capexCorrection
dfins.at['CC','capcost'] = dfins.at['CC','capcost_total']/dfins.at['CC','lifetime']*capexCorrection
dfins.at['CO2liquidification','capcost'] = dfins.at['CO2liquidification','capcost_total']/dfins.at['CO2liquidification','lifetime']*capexCorrection
dfins.at['CO2_L_store','capcost'] = dfins.at['CO2_L_store','capcost_total']/dfins.at['CO2_L_store','lifetime']*capexCorrection
dfins.at['H2compression','capcost'] = dfins.at['H2compression','capcost_total']/dfins.at['H2compression','lifetime']*capexCorrection
dfins.at['H2_HP_store','capcost'] = dfins.at['H2_HP_store','capcost_total']/dfins.at['H2_HP_store','lifetime']*capexCorrection
dfins.at['O2compression','capcost'] = dfins.at['O2compression','capcost_total']/dfins.at['O2compression','lifetime']*capexCorrection
dfins.at['electrolyzer','capcost'] = dfins.at['electrolyzer','capcost_total']/dfins.at['electrolyzer','lifetime']*capexCorrection
dfins.at['CH3OHcatalysis','capcost'] = dfins.at['CH3OHcatalysis','capcost_total']/dfins.at['CH3OHcatalysis','lifetime']*capexCorrection
dfins.at['CH4catalysis','capcost'] = dfins.at['CH4catalysis','capcost_total']/dfins.at['CH4catalysis','lifetime']*capexCorrection
dfins.at['batstore','capcost'] = dfins.at['batstore','capcost_total']/dfins.at['batstore','lifetime']*capexCorrection
dfins.at['heatstore','capcost'] = dfins.at['heatstore','capcost_total']/dfins.at['heatstore','lifetime']*capexCorrection

#%% === construct PyPSA Energy Hub Model ===
dfinsOrig = dfins.copy(deep=True) # deep copy original dataframe as starting point for variations
# == run 
for name_varia in tqdm(['varyCH4_X', 'varyCH3OH_X', 'varyCO2certificate_X']):
    
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    path = None #absolute path if you don't want to save things to the current folder
    name_folder = name_varia + '_' + timestamp
            
    _, __, N_varia = getVariationInput(name_varia, 0, dfins, name_folder, path)

    for i_varia in tqdm(range(N_varia)):
        dfins, priceFactor, _ = getVariationInput(name_varia, i_varia, dfinsOrig)

        # === pyPSA network basic construction
        n = pypsa.Network(snapshots = snapshots, override_component_attrs=extendComponentAttrs())

        #limit the reserves
        n.total_reserve_pos= 100
        n.total_reserve_neg= 100
        nScenarios = len(p_scenarios)
        print(f"{nScenarios} Szenarios werden erstellt ...")
        s = pd.Index([f"-S{s:04d}" for s in range(nScenarios)])
        n.scenarios = s
        n.p_scenarios = p_scenarios
        
        if priceFactor == -1:
            helper = scaleUsingsinh(np.hstack(np.array(preds["Da"][0])),np.mean(np.hstack(np.array(preds["Da"][0]))))
        else:
            helper = scaleUsingsinh(np.hstack(np.array(preds["Da"][0])), priceFactor)
            
        helper2 = preds["Da"][0].copy(deep=True)
        helper2.loc[:,helper2.keys()[0]] = helper
        n.da = helper2

        helper2 = scenarios["aFRR_pos"].copy(deep=True)
        helper2.loc[:,helper2.keys()[0]] = helper
        n.aFRR_pos = helper2.clip(0)
        
        helper2 = scenarios["aFRR_neg"].copy(deep=True)
        helper2.loc[:,helper2.keys()[0]] = helper
        n.aFRR_neg = helper2.clip(0)
        
        Q_fern = scenarios["Q_fern"].clip(lower=0,upper=29.605900) # maximum value of dataset

        priceLevels = {}
        priceLevels["Da"] = n.da
        
        helper2 = preds["aFRR_pos"][0].copy(deep=True)
        helper2.loc[:,helper2.keys()[0]] = helper
        priceLevels["aFRR_pos"] = helper2.clip(0) # no negative prices
        
        helper2 = preds["aFRR_neg"][0].copy(deep=True)
        helper2.loc[:,helper2.keys()[0]] = helper
        priceLevels["aFRR_neg"] = helper2.clip(0) # no negative prices
        
        helper2 = scenarios['Da'].copy(deep=True)
        helper2.loc[:,helper2.keys()[0]] = helper
        dagen = helper2
        n.priceLevels = pd.concat(priceLevels,1)

        # === Add buses ===
        # general buses
        n.madd('Bus', 'buselec'+s,     carrier = 'electric power [MW]')
        n.madd('Bus', 'busheat_DHT'+s, carrier = 'heat at DH supplyT [MW]')
        n.madd('Bus', 'busheat_60'+s,  carrier = 'heat at 60 C [MW]')
        n.madd('Bus', 'busCO2_L'+s,    carrier = 'liquid CO2 [t/h]')
        n.madd('Bus', 'busCO2_G'+s,    carrier = 'gaseous CO2 [t/h]')
        n.madd('Bus', 'busH2_LP'+s,    carrier = 'H2 low pressure [MW]') #30 bar
        n.madd('Bus', 'busH2_HP'+s,    carrier = 'H2 high pressure[MW]') #250 bar
        n.madd('Bus', 'busO2_LP'+s,    carrier = 'O2 low pressure [t/h]') #leaving electrolysis
        n.madd('Bus', 'busO2_HP'+s,    carrier = 'O2 high pressure [t/h]') #to market
        n.madd('Bus', 'busCH3OH'+s,    carrier = 'CH3OH [MW]')
        n.madd('Bus', 'busCH4'+s,      carrier = 'CH4 [MW]')

        # buses for storage with limited transfer capacity (not only storage capacity limited, also flow in/out of storage)
        n.madd('Bus', 'buselecstore'+s,         carrier = 'electric power [MW]')
        n.madd('Bus', 'busheatstore_DHT'+s,     carrier = 'heat at DH supplyT [MW]')

        # buses associated with cogeneration plant
        n.madd('Bus', 'busCHP_fuel'+s,   carrier = 'CHP fuel [MW]')
        n.madd('Bus', 'busCHP_load'+s,   carrier = 'CHP firing load [MW]')
        n.madd('Bus', 'busCHP_loss'+s,   carrier = 'CHP loss [MW]')
        n.madd('Bus', 'busCHP_CO2flue'+s,carrier = 'CHP CO2 in flue gas [t/h]') #before flue gas cleaning
        n.madd('Bus', 'busCHP_CO2exh'+s, carrier = 'CHP CO2 to exhaust [t/h]') #behind flue gas cleaning

        # === add CHP ===
        #source = fuel input
        n.madd('Generator', 'CHP_fuelinput'+s, bus = 'busCHP_fuel'+s, carrier = 'CHP fuel [MW]',
               p_nom = dfins.at['CHP','nomload'], 
               p_max_pu = dfins.at['CHP','maxload'],
               p_min_pu = dfins.at['CHP','minload'],
        #       r_pos_max=n.total_reserve_pos,r_neg_max=n.total_reserve_neg,
               ramp_limit_up = dfins.at['CHP','ramplim'],
               ramp_limit_down = dfins.at['CHP','ramplim'],
               committable = dfins.at['CHP','commitable'],
               p_nom_extendable = dfins.at['CHP','extendable'],
               marginal_cost=dfins.at['CHP','margcost']
              )

        #links = combustion process
        n.madd('Link', 'CHP_combustion'+s, bus0 = 'busCHP_fuel'+s, bus1 = 'busCHP_load'+s, 
               bus2 = 'busCHP_CO2flue'+s, 
               efficiency=1, 
               efficiency2=dfins.at['CHP', 'flueCO2'], 
               p_nom = dfins.at['CHP','nomload']
              )

        #links = steam generation and utility processes
        #cannot be easily achieved with a multibus link, because of coupled variable efficiencies -> multiple links and extra_functionality
        n.madd('Link', 'CHP_elec'+s, bus0 = 'busCHP_load'+s, bus1 = 'buselec'+s, 
               efficiency = 1, ramp_limit_up = 1, ramp_limit_down = 1, p_min_pu = 0, p_max_pu = 1,
               p_nom=dfins.at['CHP','nomelec']
              )
        n.madd('Link', 'CHP_heat'+s, bus0 = 'busCHP_load'+s, bus1 = 'busheat_DHT'+s, 
               efficiency=1, p_min_pu=0, p_max_pu=1,
               p_nom = dfins.at['CHP','nomheat']
              )
        n.madd('Link', 'CHP_loss'+s, bus0 = 'busCHP_load'+s, bus1 = 'busCHP_loss'+s,
               efficiency = 1, p_min_pu=0, p_max_pu=1, 
               p_nom = dfins.at['CHP','nomload']
              )
        #sink for losses
        n.madd('Generator', 'CHP_losssink'+s, bus = 'busCHP_loss'+s, 
               p_nom=bigM, p_max_pu=0, p_min_pu=-1, marginal_cost=0
              )


        # === add CO2 separator and exhaust ===

        #add Carbon Capture
        n.madd('Link', 'CHP_CC'+s, bus0='busCHP_CO2flue'+s, 
               bus1 = 'busCO2_G'+s, efficiency = dfins.at['CC','eff'],
               bus2 = 'buselec'+s, efficiency2 = dfins.at['CC','elecout'], 
               bus3 = 'busheat_DHT'+s, efficiency3 = dfins.at['CC','heatDHTout'],
               bus4 = 'busCHP_CO2exh'+s, efficiency4 = 1 - dfins.at['CC','eff'],
               bus5 = 'busheat_60'+s, efficiency5 = dfins.at['CC','heat60out'],
               p_nom = dfins.at['CC','nomload'],
               p_min_pu=0, p_max_pu=1, 
               capital_cost= dfins.at['CC','capcost'],
               p_nom_extendable = dfins.at['CC','extendable']
              )
        #remaining CO2 goes directly to exhaust
        n.madd('Link', 'CHP_noCC'+s, bus0 = 'busCHP_CO2flue'+s, bus1 = 'busCHP_CO2exh'+s, 
               efficiency = 1, p_nom = bigM, p_min_pu = 0, p_max_pu = 1
              )
        #from exhaust CO2 is emitted, for which emission rights need to be paid
        n.madd('Generator', 'CHP_exhaust'+s, bus = 'busCHP_CO2exh'+s, 
               p_nom = bigM, p_max_pu = 0, p_min_pu = -1, 
               marginal_cost = dfins.at['CO2_exhaust','margcost']
              )

        # === add CO2 liquidification, gasification and storage ===
        n.madd('Link', 'CO2liquidification'+s, bus0 = 'busCO2_G'+s, 
               bus1 = 'busCO2_L'+s, efficiency = dfins.at['CO2liquidification','eff'], 
               bus2 = 'buselec'+s, efficiency2 = dfins.at['CO2liquidification','elecout'], 
               bus3 = 'busheat_DHT'+s, efficiency3 = dfins.at['CO2liquidification','heatDHTout'], 
               p_nom = dfins.at['CO2liquidification','nomload'], 
               p_min_pu = 0, p_max_pu = 1, 
               capital_cost = dfins.at['CO2liquidification','capcost'], 
               p_nom_extendable = dfins.at['CO2liquidification','extendable']
              )


        n.madd('Link', 'CO2gasification'+s, bus0 = 'busCO2_L'+s, 
               bus1 = 'busCO2_G'+s,
               p_min_pu = 0, p_max_pu = 1,
               p_nom = smallM 
              )

        # note that store directly coupled to main bus does not have limitation on max flow, to get this you need an additional bus and (dis)charge links. 
        n.madd('Store', 'CO2_L_store'+s, bus = 'busCO2_L'+s,
               standing_loss = dfins.at['CO2_L_store','standloss'],
               e_nom = dfins.at['CO2_L_store','nomsize'],
               e_initial = dfins.at['CO2_L_store','e_initial'],
               e_cyclic = dfins.at['CO2_L_store','e_cyclic'],
               capital_cost = dfins.at['CO2_L_store','capcost'],
               e_nom_extendable = dfins.at['CO2_L_store','extendable']
              )

        # === add CO2 market ===
        n.madd('Generator','CO2_L_market'+s, bus = 'busCO2_L'+s, 
               p_nom = bigM, p_max_pu = 1*dfins.at['CO2_L_market','buy'], p_min_pu = -1*dfins.at['CO2_L_market','sell'],
               marginal_cost = dfins.at['CO2_L_market','margcost']
              )

        # === add resistive heater ===
        n.madd('Link', 'heatrod'+s, bus0 = 'buselec'+s, bus1 = 'busheat_DHT'+s, 
               efficiency = dfins.at['heatrod','eff'], 
               p_nom = dfins.at['heatrod','nomload'], 
               p_min_pu = 0, p_max_pu = 1, 
               capital_cost = dfins.at['heatrod','capcost'], 
               p_nom_extendable = dfins.at['heatrod','extendable']
              )

        # === add heat pumps ===
        n.madd('Link', 'heatpump_25_120'+s, bus0 = 'buselec'+s, bus1 = 'busheat_DHT'+s, 
               efficiency = dfins.at['heatpump_25_120','eff'], 
               p_nom = dfins.at['heatpump_25_120','nomload'], 
               p_min_pu = 0, p_max_pu = 1, 
               capital_cost = dfins.at['heatpump_25_120','capcost'], 
               p_nom_extendable = dfins.at['heatpump_25_120','extendable']
              )

        n.madd('Link', 'heatpump_60_120'+s, bus0 = 'buselec'+s, 
               bus1 = 'busheat_DHT'+s, efficiency = dfins.at['heatpump_60_120','eff'], 
               bus2 = 'busheat_60'+s, efficiency2 = -(dfins.at['heatpump_60_120','eff']-1),
               p_nom = dfins.at['heatpump_60_120','nomload'], 
               p_min_pu = 0, p_max_pu = 1, 
               capital_cost = dfins.at['heatpump_60_120','capcost'], 
               p_nom_extendable = dfins.at['heatpump_60_120','extendable']
              )

        # === add sink for 60C temperature level
        n.madd('Generator', '60C_sink'+s, bus = 'busheat_60'+s, 
               p_nom=bigM, p_max_pu=0, p_min_pu=-1, marginal_cost=0
              )
        # === add BESS ===
        #link for charging and discharging
        n.madd('Link', 'batcharge'+s, bus0 = 'buselec'+s, bus1 = 'buselecstore'+s, 
               efficiency = dfins.at['batcharge','eff'],
               p_nom = dfins.at['batcharge','nomload'], 
               p_min_pu = -1, p_max_pu = 1,
               capital_cost = dfins.at['batcharge','capcost'],
               p_nom_extendable = dfins.at['batcharge','extendable']
              )

        n.madd('Store', 'batstore'+s, bus = 'buselecstore'+s,
               standing_loss = dfins.at['batstore','standloss'],
               e_nom = dfins.at['batstore','nomsize'],
               e_initial = dfins.at['batstore','e_initial'],
               e_cyclic = dfins.at['batstore','e_cyclic'],
               capital_cost = dfins.at['batstore','capcost'],
               e_nom_extendable = dfins.at['batstore','extendable']
              )


        # n.madd('StorageUnit','bat'+s, bus='buselecstore'+s, state_of_charge_initial=0,p_nom=0.1,standing_loss = standinglossbat,p_min_pu=-1,
        #        r_pos_max=n.total_reserve_pos,r_neg_max=n.total_reserve_neg)#, capital_cost=capc['Bat'][u], p_nom_extendable=True klappt mit r_pos nicht 

        # === add thermal storage ===
        n.madd('Link', 'heatcharge'+s, bus0 = 'busheat_DHT'+s, bus1 = 'busheatstore_DHT'+s, 
               efficiency = dfins.at['heatcharge','eff'],
               p_nom = dfins.at['heatcharge','nomload'], 
               p_min_pu = -1, p_max_pu = 1,
               capital_cost = dfins.at['heatcharge','capcost'],
               p_nom_extendable = dfins.at['heatcharge','extendable']
              )

        n.madd('Store', 'heatstore'+s, bus = 'busheatstore_DHT'+s,
               standing_loss = dfins.at['heatstore','standloss'],
               e_nom = dfins.at['heatstore','nomsize'],
               e_initial = dfins.at['heatstore','e_initial'],
               e_cyclic = dfins.at['heatstore','e_cyclic'],
               capital_cost = dfins.at['heatstore','capcost'],
               e_nom_extendable = dfins.at['heatstore','extendable']
              )

        # === add thermal load ===
        n.madd('Load', 'loadheat'+s, bus = 'busheat_DHT'+s, 
               p_set = Q_fern, carrier = 'heat at DH supplyT')

        # === add penalty generator so that network stays feasible also if heat load is not covered
        n.madd('Generator','penalty'+s, bus='busheat_DHT'+s, p_max_pu=1, p_min_pu=0, p_nom= bigM, marginal_cost = bigM)

        # === add day ahead market ===
        dabuy = dagen.copy(deep=True) + dfins.at['DAmarket','buyfee']
        dabuy.rename(columns={'dagensell-S0000': "dagenbuy-S0000"}, inplace=True)

        n.madd('Generator', 'dagensell'+s, bus='buselec'+s, 
               p_max_pu = 0, p_min_pu = -1 * (dfins.at['DAmarket','sell']),
               p_nom = dfins.at['trafo','nomload'],
               marginal_cost = dagen
              )

        n.madd('Generator', 'dagenbuy'+s, bus='buselec'+s, 
               p_max_pu = 1 * (dfins.at['DAmarket','buy']), p_min_pu = 0, 
               p_nom= dfins.at['trafo','nomload'],
               marginal_cost = dabuy
              )

        # === add H2 compression ===
        n.madd('Link', 'H2compression'+s, bus0 = 'busH2_LP'+s, 
               bus1 = 'busH2_HP'+s, efficiency = 1,
               bus2 = 'buselec'+s, efficiency2 = dfins.at['H2compression','elecout'],
               p_nom = dfins.at['H2compression','nomload'],
               p_min_pu=0, p_max_pu=1, 
               p_nom_extendable = dfins.at['H2compression','extendable'],
               capital_cost = dfins.at['H2compression','capcost']
              )

        # === add H2 expansion ===
        n.madd('Link', 'H2expansion'+s, bus0 = 'busH2_HP'+s, 
               bus1 = 'busH2_LP'+s,
               p_min_pu=0, p_max_pu=1,
               p_nom = bigM
              )

        # === add compressed H2 storage ===
        n.madd('Store', 'H2_HP_store'+s, bus = 'busH2_HP'+s,
               standing_loss = dfins.at['H2_HP_store','standloss'],
               e_nom = dfins.at['H2_HP_store','nomsize'],
               e_initial = dfins.at['H2_HP_store','e_initial'],
               e_cyclic = dfins.at['H2_HP_store','e_cyclic'],
               capital_cost = dfins.at['H2_HP_store','capcost'],
               e_nom_extendable = dfins.at['H2_HP_store','extendable']
              )

        # === add compressed H2 market ===
        n.madd('Generator','H2_HP_market'+s, bus = 'busH2_HP'+s, 
               p_nom = bigM, p_max_pu = 1*dfins.at['H2_HP_market','buy'], p_min_pu = -1*dfins.at['H2_HP_market','sell'],
               marginal_cost = dfins.at['H2_HP_market','margcost']
              )

        # === add O2 compression ===
        n.madd('Link', 'O2compression'+s, bus0 = 'busO2_LP'+s, 
               bus1 = 'busO2_HP'+s, efficiency = 1,
               bus2 = 'buselec'+s, efficiency2 = dfins.at['O2compression','elecout'],
               p_nom = dfins.at['O2compression','nomload'],
               p_min_pu=0, p_max_pu=1, 
               p_nom_extendable = dfins.at['O2compression','extendable'],
               capital_cost = dfins.at['O2compression','capcost']
              )

        # === add O2 expansion ===
    #    n.madd('Link', 'O2expansion'+s, bus0 = 'busO2_HP'+s, 
    #           bus1 = 'busO2_LP'+s,
    #           p_min_pu=0, p_max_pu=1,
    #           p_nom = bigM
    #          )

        # === add sink for O2 ===

        n.madd('Generator', 'O2_sink'+s, bus = 'busO2_LP'+s, 
               p_nom=bigM, p_max_pu=0, p_min_pu=-1, marginal_cost=0
              )


        # === add compressed O2 market ===
        n.madd('Generator','O2_HP_market'+s, bus = 'busO2_HP'+s, 
               p_nom = bigM, p_max_pu = 1*dfins.at['O2_HP_market','buy'], p_min_pu = -1*dfins.at['O2_HP_market','sell'],
               marginal_cost = dfins.at['O2_HP_market','margcost']
              )

        # === add electrolyzer ===
        n.madd('Link', 'electrolyzer'+s, bus0 = 'buselec'+s, 
               bus1 = 'busH2_LP'+s, efficiency = dfins.at['electrolyzer','eff'],
               bus2 = 'busheat_60'+s, efficiency2 = dfins.at['electrolyzer','heat60out'],
               bus3 = 'busO2_LP'+s, efficiency3 = dfins.at['electrolyzer', 'O2out'],
               p_nom = dfins.at['electrolyzer','nomload'],
               p_min_pu = 0, p_max_pu = 1, 
               capital_cost = dfins.at['electrolyzer','capcost'],
               p_nom_extendable = dfins.at['electrolyzer','extendable']
              )

        # === add CH3OHcatalysis ===
        n.madd('Link', 'CH3OHcatalysis'+s, bus0 = 'busH2_LP'+s, 
               bus1 = 'busCH3OH'+s, efficiency = dfins.at['CH3OHcatalysis','eff'],
               bus2 = 'busCO2_G'+s, efficiency2 = dfins.at['CH3OHcatalysis','CO2out'],
               bus3 = 'buselec'+s, efficiency3 = dfins.at['CH3OHcatalysis','elecout'],
               bus4 = 'busheat_DHT'+s, efficiency4 = dfins.at['CH3OHcatalysis','heatDHTout'],
               p_nom = dfins.at['CH3OHcatalysis','nomload'],
               p_min_pu=0,p_max_pu=1,
               capital_cost = dfins.at['CH3OHcatalysis','capcost'],
               p_nom_extendable = dfins.at['CH3OHcatalysis','extendable']
              )

        # === add CH3OH market ===
        n.madd('Generator', 'CH3OH'+s, bus = 'busCH3OH'+s, 
               p_nom = bigM, p_max_pu = 1*dfins.at['CH3OHmarket','buy'], p_min_pu = -1*dfins.at['CH3OHmarket','sell'],
               marginal_cost = dfins.at['CH3OHmarket','margcost']
              ) 

        # === add CH4catalysis ===
        n.madd('Link', 'CH4catalysis'+s, bus0 = 'busH2_LP'+s, 
               bus1 = 'busCH4'+s, efficiency = dfins.at['CH4catalysis','eff'],
               bus2 = 'busCO2_G'+s, efficiency2 = dfins.at['CH4catalysis','CO2out'],
               bus3 = 'buselec'+s, efficiency3 = dfins.at['CH4catalysis','elecout'],
               bus4 = 'busheat_DHT'+s, efficiency4 = dfins.at['CH4catalysis','heatDHTout'],
               p_nom = dfins.at['CH4catalysis','nomload'],
               p_min_pu=0,p_max_pu=1,
               capital_cost = dfins.at['CH4catalysis','capcost'],
               p_nom_extendable = dfins.at['CH4catalysis','extendable']
              )

        # === add CH4 market ===
        n.madd('Generator', 'CH4'+s, bus = 'busCH4'+s, 
               p_nom = bigM, p_max_pu = 1*dfins.at['CH4market','buy'], p_min_pu = -1*dfins.at['CH4market','sell'],
               marginal_cost = dfins.at['CH4market','margcost']
              ) 

        n.lopf(snapshots, 
           solver_name = "gurobi", 
           pyomo=True,
           extra_postprocessing=extra_postprocessing,
           extra_functionality=extra_functionality
          )

        writeVariationOutput(name_folder, i_varia, N_varia, n, path=path)


