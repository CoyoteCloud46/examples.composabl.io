import json
import os
import random
import math
import pandas as pd

import numpy as np
import sys
from casadi import *
from scipy import interpolate

import do_mpc

import matplotlib.pyplot as plt


# time step (seconds) between state updates
Δt = 1

π = math.pi

def non_lin_mpc(noise, CrSP, Ca0, T0, Tc0):
    #constants
    F = 1 #Volumetric flow rate (m3/h)
    V = 1 #Reactor volume (m3)
    k0 = 34930800 #Pre-exponential nonthermal factor (1/h)
    E = 11843 #Activation energy per mole (kcal/kmol)
    R = 1.985875 #Boltzmann's ideal gas constant (kcal/(kmol·K))
    ΔH = -5960 #Heat of reaction per mole kcal/kmol
    phoCp = 500 #Density multiplied by heat capacity (kcal/(m3·K))
    UA = 150 #Overall heat transfer coefficient multiplied by tank area (kcal/(K·h))
    Cafin = 10 #kmol/m3
    Tf = 298.2 #K

    #Ca0 = 8.5698  # kmol/m3
    #T0 = 311.2639  # K
    #Tc0 = 292  # K

    #MPC MODEL
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)
    # States struct (optimization variables):
    tau = 3
    Kp = 0.65 #0.75

    Ca = model.set_variable(var_type='_x', var_name='Ca', shape=(1,1)) #Concentration
    T = model.set_variable(var_type='_x', var_name='T', shape=(1,1)) #Temperature

    # define measurements:
    model.set_meas('Ca', Ca, meas_noise=True)
    model.set_meas('T', T, meas_noise=True)

    # Input struct (optimization variables):
    Tc = model.set_variable(var_type='_u', var_name='Tc') #cooling liquid temperature

    # Uncertain parameters:
    Caf = model.set_variable(var_type='_tvp', var_name='Caf')
    Tref = model.set_variable(var_type='_tvp', var_name='Tref')

    model.set_rhs('T',  (-(T - T0) + Kp * (Tc - Tc0))/tau )
    model.set_rhs('Ca', (F/V * (Cafin - Ca)) - (k0 * exp(-E/(R*T))*Ca)  )
    #model.set_rhs('T', (F/V *(Tf-T)) - ((ΔH/phoCp)*(k0 * exp(-E/(R*T))*Ca)) - ((UA /(phoCp*V)) *(T-Tc)) )

    # Build the model
    model.setup()

    #CONTROLLER
    mpc = do_mpc.controller.MPC(model)
    setup_mpc = {
        'n_horizon': 20,
        'n_robust': 1,
        'open_loop': 0,
        't_step': Δt,
        'store_full_solution': True
    }

    mpc.set_param(**setup_mpc)
    surpress_ipopt = {'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0}
    mpc.set_param(nlpsol_opts = surpress_ipopt)

    mpc.scaling['_x', 'T'] = 100
    mpc.scaling['_u', 'Tc'] = 100

    #OBJECTIVE
    _x = model.x
    _tvp = model.tvp
    _u = model.u

    mterm = ((_x['Ca'] - CrSP))**2 # terminal cost
    lterm = ((_x['Ca'] - CrSP))**2 # stage cost

    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.set_rterm(Tc=1.5 * 1) # input penalty 1e-2 , 1000

    #Constraints
    # bounds of the states
    mpc.bounds['lower', '_x', 'Ca'] = 0.1
    mpc.bounds['upper', '_x', 'Ca'] = 12

    mpc.bounds['upper', '_x', 'T'] = 400 #
    mpc.bounds['lower', '_x', 'T'] = 100

    # lower bounds of the inputs
    mpc.bounds['lower', '_u', 'Tc'] = 273 #273

    # upper bounds of the inputs
    mpc.bounds['upper', '_u', 'Tc'] = 322

    #UNCERTAIN VALUES
    # in optimizer configuration:
    tvp_temp_1 = mpc.get_tvp_template()
    tvp_temp_1['_tvp', :] = np.array([8.5698])

    tvp_temp_2 = mpc.get_tvp_template()
    tvp_temp_2['_tvp', :] = np.array([2])

    tvp_temp_3 = mpc.get_tvp_template()
    tvp_temp_3['_tvp', :] = np.array([2])


    def tvp_fun(t_now):
        p1 = 22 #22 10
        p2 = 74 #74 36
        time = 90
        ceq = [8.57,6.9275,5.2850,3.6425,2]
        teq = [311.2612,327.9968,341.1084,354.7246,373.1311]
        C = interpolate.interp1d([0,p1,p2,time], [8.57,8.57,2,2])
        T_ = interpolate.interp1d([0,p1,p2,time], [311.2612,311.2612,373.1311,373.1311])

        if t_now < p1:
            return tvp_temp_1
        elif t_now >= p1 and t_now < p2:
            #y = -0.2527*t_now + 11.097
            y = float(C(k))
            tvp_temp_3['_tvp', :] = np.array([y])
            return tvp_temp_3
        else:
            return tvp_temp_2

    mpc.set_tvp_fun(tvp_fun)

    mpc.setup()

    #ESTIMATOR
    estimator = do_mpc.estimator.StateFeedback(model)

    #SIMULATOR
    simulator = do_mpc.simulator.Simulator(model)
    params_simulator = {
        't_step': Δt
    }

    simulator.set_param(**params_simulator)


    #uncertain parameters
    p_num = simulator.get_p_template()
    tvp_num = simulator.get_tvp_template()

    # function for time-varying parameters
    def tvp_fun(t_now):
        return tvp_num

    # uncertain parameters

    def p_fun(t_now):
        return p_num

    simulator.set_tvp_fun(tvp_fun)
    simulator.set_p_fun(p_fun)


    simulator.setup()

    # Set the initial state of mpc, simulator and estimator:
    x0 = simulator.x0
    x0['Ca'] = Ca0
    x0['T'] = T0

    u0 = simulator.u0
    u0['Tc'] = Tc0

    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0

    mpc.u0 = u0
    simulator.u0 = u0
    estimator.u0 = u0


    mpc.set_initial_guess()

    #Simulate N steps
    u0_old = 0
    time = 1
    for k in range(time):
        if k > 1:
            u0_old = u0[0][0]


        u0 = mpc.make_step(x0)
        #fix from -10 to 10
        if k > 1:
            if u0[0][0] - u0_old > 10:
                u0 = np.array([[u0_old + 10]])
            elif u0[0][0] - u0_old < -10:
                u0 = np.array([[u0_old - 10]])
        else:
            if u0[0][0] - Tc0 >= 10:
                u0 = np.array([[Tc0 + 10]])
            elif u0[0][0] - Tc0 <= -10:
                u0 = np.array([[Tc0 - 10]])

        #Add Noise
        error_var = noise
        σ_max1 = error_var * (8.5698 -2)
        σ_max2 = error_var * ( 373.1311 - 311.2612)
        mu = 0
        v0 = np.array([mu + σ_max1* np.random.randn(1, 1)[0],mu + σ_max2* np.random.randn(1, 1)[0]])

        y_next = simulator.make_step(u0,v0=v0) # MPC

        #get all state values
        state_ops = y_next.reshape((1,2))

        #Benchmark
        p1 = 22 #22 10
        p2 = 74 #74 36
        ceq = [8.57,6.9275,5.2850,3.6425,2]
        teq = [311.2612,327.9968,341.1084,354.7246,373.1311]
        C = interpolate.interp1d([0,p1,p2,time], [8.57,8.57,2,2])
        T_ = interpolate.interp1d([0,p1,p2,time], [311.2612,311.2612,373.1311,373.1311])
        if k < p1:
            Cref = 8.5698
            Tref = 311.2612
        elif k >= p1 and k < p2:
            y = float(C(k))
            y2 = float(T_(k))
            Cref = y
            Tref = y2
        else:
            Cref = 2
            Tref = 373.1311

        x0 = estimator.make_step(y_next) # MPC

    return u0




