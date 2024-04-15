import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from composabl import Agent, Runtime, Scenario, Sensor, Skill
from sensors import sensors
from config import config
from scenarios import Navigation_scenarios
from teacher import (AlignmentTeacher, SelectorTeacher, SpeedControlTeacher, StabilizationTeacher,NavigationTeacher)

PATH: str = os.path.dirname(os.path.realpath(__file__))
PATH_HISTORY: str = f"{PATH}/history"
PATH_CHECKPOINTS : str = f"{PATH}/checkpoints"

def run_agent():
    # delete old history files
    Stabilization_skill = Skill("Stabilization", StabilizationTeacher)
    Navigation_skill = Skill("Navigation", NavigationTeacher)
    Alignment_skill = Skill("Alignment", AlignmentTeacher)
    SpeedControl_skill = Skill("SpeedControl", SpeedControlTeacher)
    selector_skill = Skill("selector", SelectorTeacher)

    for scenario_dict in Navigation_scenarios:
        scenario = Scenario(scenario_dict)
        Navigation_skill.add_scenario(scenario)
        Alignment_skill.add_scenario(scenario)
        SpeedControl_skill.add_scenario(scenario)
        Stabilization_skill.add_scenario(scenario)
        selector_skill.add_scenario(scenario)

    runtime = Runtime(config)
    agent = Agent()
    agent.add_sensors(sensors)

    agent.add_skill(Navigation_skill)
    agent.add_skill(Stabilization_skill)
    agent.add_skill(Alignment_skill)
    agent.add_skill(SpeedControl_skill)
    agent.add_selector_skill(selector_skill, [Stabilization_skill, Alignment_skill], fixed_order=False, fixed_order_repeat=False)

    # Load a pre-trained agent
    try:
        if len(os.listdir(PATH_CHECKPOINTS)) > 0:
            agent.load(PATH_CHECKPOINTS)
    except Exception:
        print("|-- No checkpoints found. Training from scratch...")

    # Start training the agent
    runtime.train(agent, train_iters=2)

    # Save the trained agent
    agent.export(PATH_CHECKPOINTS)


if __name__ == "__main__":
    run_agent()