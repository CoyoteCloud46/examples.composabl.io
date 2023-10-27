import os

from composabl import Agent, Runtime, Scenario, Sensor, Skill

from teacher import CSTRTeacher, SS1Teacher, SS2Teacher, TransitionTeacher
from perceptors import perceptors

from cstr.external_sim.sim import CSTREnv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

license_key = os.environ["COMPOSABL_KEY"]

from composabl import Controller

class ProgrammedSelector(Controller):
    def __init__(self):
        self.counter = 0
        
    def compute_action(self, obs):
        if self.counter < 22:
            action = [0]
        elif self.counter < 74 : #transition
            action = [1]
        else:
            action = [2]

        self.counter += 1
            
        return action

    def transform_obs(self, obs):
        return obs

    def filtered_observation_space(self):
        return ['T', 'Tc', 'Ca', 'Cref', 'Tref']
    
    def compute_success_criteria(self, transformed_obs, action):
        if self.counter > 100:
            return True

    def compute_termination(self, transformed_obs, action):
        return False


T = Sensor("T", "")
Tc = Sensor("Tc", "")
Ca = Sensor("Ca", "")
Cref = Sensor("Cref", "")
Tref = Sensor("Tref", "")

sensors = [T, Tc, Ca, Cref, Tref]

# Cref_signal is a configuration variable for Concentration and Temperature setpoints
ss1_scenarios = [
    {
        "Cref_signal": "ss1"
    }
]

ss2_scenarios = [
    {
        "Cref_signal": "ss2"
    }
]

transition_scenarios = [
    {
        "Cref_signal": "transition"
    }
]

selector_scenarios = [
    {
        "Cref_signal": "complete"
    }
]

ss1_skill = Skill("ss1", SS1Teacher, trainable=True)
for scenario_dict in ss1_scenarios:
    ss1_skill.add_scenario(Scenario(scenario_dict))

ss2_skill = Skill("ss2", SS2Teacher, trainable=True)
for scenario_dict in ss2_scenarios:
    ss2_skill.add_scenario(Scenario(scenario_dict))

transition_skill = Skill("transition", TransitionTeacher, trainable=True)
for scenario_dict in transition_scenarios:
    transition_skill.add_scenario(Scenario(scenario_dict))

selector_skill = Skill("selector", ProgrammedSelector, trainable=False)
for scenario_dict in selector_scenarios:
    selector_skill.add_scenario(Scenario(scenario_dict))


config = {
    "license": license_key,
    "target": {
        "docker": {
            "image": "composabl/sim-cstr:latest"
        }
    },
    "env": {
        "name": "sim-cstr",
    },
    "runtime": {
        "ray": {
            "workers": 1
        }
    }
}

runtime = Runtime(config)
agent = Agent(runtime, config)
agent.add_sensors(sensors)
agent.add_perceptors(perceptors)

agent.add_skill(ss1_skill)
agent.add_skill(ss2_skill)
agent.add_skill(transition_skill)
agent.add_selector_skill(selector_skill, [ss1_skill, transition_skill, ss2_skill], fixed_order=False, fixed_order_repeat=False)

checkpoint_path = './cstr/multiple_skills_perceptor/saved_agents/'

#load agent
agent.load(checkpoint_path)

#save agent
trained_agent = agent.prepare()

sim = CSTREnv()
sim.scenario = Scenario({
        "Cref_signal": "complete",
        "noise_percentage": 0.05
    })

df = pd.DataFrame()

for i in range(30):
    # Inference
    obs, info = sim.reset()
    for i in range(90):
        action = trained_agent.execute(obs)
        action = np.array((action[0]+10)/20)
        obs, reward, done, truncated, info = sim.step(action)
        df_temp = pd.DataFrame(columns=['T','Tc','Ca','Cref','Tref','time'],data=[list(obs) + [i]])
        df = pd.concat([df, df_temp])

        if done:
            break

# calculate error
df['error_temp'] = (df['T'] - df['Tref'])**2
df['error_conc'] = (df['Ca'] - df['Cref'])**2
rmsT = round(np.sqrt(np.mean(df['error_temp'])),2)
rmsCa = round(np.sqrt(np.mean(df['error_conc'])),2)
print('RMS T: ', rmsT)
print('RMS Ca: ', rmsCa)

plt.subplot(2,1,1)
Tr_list_ = df.set_index('time')['T']
Tref_list = df.set_index('time')['Tref'][:90]
min_Tr = [min(np.array(Tr_list_[i])) for i in range(90)]
max_Tr = [max(np.array(Tr_list_[i])) for i in range(90)]
mean_Tr = [np.mean(np.array(Tr_list_[i])) for i in range(90)]
plt.fill_between([i for i in range(90)] , min_Tr , max_Tr, alpha = 0.2)
plt.plot([i for i in range(90)],Tref_list,'k--',lw=2,label=r'$T_{sp}$')
plt.plot([i for i in range(90)],mean_Tr,'b.-',lw=1,label=r'$T_{sp}$')
plt.plot([i for i in range(90)],[400 for x in range(90)],'r--',lw=1)
plt.ylabel('Temperature')
plt.title('Benchmarks' + f" (RMS T: {rmsT} , RMS Ca: {rmsCa})")

plt.subplot(2,1,2)
Cr_list_ = df.set_index('time')['Ca']
Cref_list = df.set_index('time')['Cref'][:90]
min_Cr = [min(np.array(Cr_list_[i])) for i in range(90)]
max_Cr = [max(np.array(Cr_list_[i])) for i in range(90)]
mean_Cr = [np.mean(np.array(Cr_list_[i])) for i in range(90)]
plt.fill_between([i for i in range(90)] , min_Cr , max_Cr, alpha = 0.2)
plt.plot([i for i in range(90)],Cref_list,'k--',lw=2,label=r'$C_{sp}$')
plt.plot([i for i in range(90)],mean_Cr,'b.-',lw=1,label=r'$C_{sp}$')
plt.ylabel('Concentration')

plt.savefig('./cstr/multiple_skills_perceptor/benchmark_figure.png')