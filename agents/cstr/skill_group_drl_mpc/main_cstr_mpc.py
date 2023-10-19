from composabl.core import Agent, Skill, Sensor, Scenario
from composabl.ray import Runtime
from .cstr_mpc_teacher import CSTRTeacher

from composabl import Controller

import os
import numpy as np

os.environ["COMPOSABL_EULA_AGREED"] = "1"
license_key = os.environ["COMPOSABL_KEY"]
    
    
def start():
    T = Sensor("T", "")
    Tc = Sensor("Tc", "")
    Ca = Sensor("Ca", "")
    Cref = Sensor("Cref", "")
    Tref = Sensor("Tref", "")

    sensors = [T, Tc, Ca, Cref, Tref]

    # Cref_signal is a configuration variable for Concentration and Temperature setpoints
    control_scenarios = [
        {
            "Cref_signal": "complete",
            "noise_percentage": 0.0
        }
    ]

    control_skill = Skill("control", CSTRTeacher, trainable=True)
    for scenario_dict in control_scenarios:
        control_skill.add_scenario(Scenario(scenario_dict))

    config = {
        "env": {
            "name": "sim-cstr", #"composabl/sim-cstr",
            "compute": "local",  # "docker", "kubernetes", "local"
            "config": {
                "address": "localhost:1337",
                #"image": "composabl/sim-cstr:latest"
            }
        },
        "license": license_key,
        "training": {}
    }
    runtime = Runtime(config)
    agent = Agent(runtime, config)
    agent.add_sensors(sensors)

    agent.add_skill(control_skill)

    agent.train(train_iters=5)