import math
import random
import numpy as np

from composabl_core.agent.scenario import Scenario
import gymnasium as gym

from rllib.whisky_business_env import WhiskeyBusinessEnv
from simulation.sim_controller import SimController
import simpy

class Env(gym.Env):
    def __init__(self):
        '''
        actions =
        observation_variables =
        '''
        config={
            "env_config": {
                "debug": False,
                "render_mode": "human"
            }
        }
        self.business_env = WhiskeyBusinessEnv(config)

        self.action_space = self.business_env.action_space
        '''self.action_space = gym.spaces.Dict({'mix_bake_decorate': gym.spaces.Discrete(3),
                                             'chip_coco_eclair_wait': gym.spaces.Discrete(3),
                                             'product_equip': gym.spaces.Discrete(3)
                             
                             })'''

        obs_space_constraints = {
            "action_mask":  { "low": 0, "high": 1},
            "observations":  {"low": 0, "high": 500},
            "dessert_cases":  {"low": 0, "high": 700},
            "dessert_prices":  {"low": 0, "high": 50}
        }

        low_list = []
        high_list = []

        for key in obs_space_constraints.keys():
            llist = [obs_space_constraints[key]["low"]]
            hlist = [obs_space_constraints[key]["high"]]
            if key == "action_mask":
                max_avail_actions = 25
                llist = llist * max_avail_actions
                hlist = hlist * max_avail_actions
            elif key == "observations":
                len_default_state = 19
                llist = llist * len_default_state
                hlist = hlist * len_default_state
            else:
                llist = llist * 9 
                hlist = hlist * 9 

            low_list += llist
            high_list += hlist

        self.cookies_price = 5
        self.cupcake_price = 7
        self.cake_price = 10
        self.cookies_demand = 30
        self.cupcake_demand = 10
        self.cake_demand = 20
        self.cookies_cost = 2
        self.cupcake_cost = 5
        self.cake_cost = 7

        self.observation_space = gym.spaces.Box(low=np.array(low_list), high=np.array(high_list))
        self.cnt = 0
        self.scenario: Scenario = None

    def process_state(self, state):
        new_state = []
        for key in state.keys():
            if key == "action_mask":
                ss = [ 1. if x == True else 0. for x in list(state[key])]
            else:
                ss = list(state[key])
            new_state += ss

        return np.array(new_state)


    def reset(self):
        # Define scenario in the simulation
        if isinstance(self.scenario, Scenario):
            sample = self.scenario.sample()

            for key in list(sample.keys()):
                setattr(self, key, sample[key])

        self.business_env.cookies_price = self.cookies_price
        self.business_env.cupcake_price = self.cupcake_price
        self.business_env.cake_price = self.cake_price
        self.business_env.cookies_demand = self.cookies_demand
        self.business_env.cupcake_demand = self.cupcake_demand
        self.business_env.cake_demand = self.cake_demand
        self.business_env.cookies_cost = self.cookies_cost
        self.business_env.cupcake_cost=  self.cupcake_cost
        self.business_env.cake_cost = self.cake_cost

        obs, info = self.business_env.reset()
        self.obs = self.process_state(obs)
        print('OBS: ', self.obs)

        return self.obs, info

    def set_scenario(self, scenario):
        self.scenario = scenario

    def step(self, action):
        #action = action[0]
        #map actions
        step_action_dict = {
            0: {#0:"wait",
                '00' :1, #"Chip_mix_cookies",
                '01' :2, #"Chip_mix_cupcakes",
                '02' :3, #"Chip_mix_cakes",
                '10' :4, #"Coco_mix_cookies",
                '11' :5, #"Coco_mix_cupcakes",
                '12' :6, #"Coco_mix_cakes",
                '20' :7, #"Eclair_mix_cookies",
                '21' :8, #"Eclair_mix_cupcakes",
                '22' :9, #"Eclair_mix_cakes"
            },
        
            1: {
                '00':10,#"Chip_bake_from_Mixer_1",
                '01':11,#"Chip_bake_from_Mixer_2",
                '02':0, 
                '10':12,#"Coco_bake_from_Mixer_1",
                '11':13,#"Coco_bake_from_Mixer_2",
                '12':0,
                '20':14,#"Eclair_bake_from_Mixer_1",
                '21':15,#"Eclair_bake_from_Mixer_2"
                '22':0
                },

            2: {
                '00':16, #"Chip_decorate_from_Oven_1",
                '01':17, #"Chip_decorate_from_Oven_2",
                '02':18, #"Chip_decorate_from_Oven_3",
                '10':19, #"Eclair_decorate_from_Oven_1",
                '11':20, #"Eclair_decorate_from_Oven_2",
                '12':21, #"Eclair_decorate_from_Oven_3",
                '20':22, #"Reese_decorate_from_Oven_1",
                '21':23, #"Reese_decorate_from_Oven_2",
                '22':24, #"Reese_decorate_from_Oven_3"
            }
        }

        #action = step_action_dict[action['mix_bake_decorate']][str(action['chip_coco_eclair_wait']) + str(action['product_equip'])]
        #print('ACTION: ', action)
        done = False
        # Increase time counter dt=1 minute
        self.cnt += 1

        self.business_env.cookies_price = self.cookies_price
        self.business_env.cupcake_price = self.cupcake_price
        self.business_env.cake_price = self.cake_price
        self.business_env.cookies_demand = self.cookies_demand
        self.business_env.cupcake_demand = self.cupcake_demand
        self.business_env.cake_demand = self.cake_demand
        self.business_env.cookies_cost = self.cookies_cost
        self.business_env.cupcake_cost=  self.cupcake_cost
        self.business_env.cake_cost = self.cake_cost

        self.obs, reward, terminate, done, info = self.business_env.step(action)
        self.obs = self.process_state(self.obs)

        return self.obs, reward, terminate, done, info

    def render(self, mode='human', close=False):
        print("render")