import asyncio
import os

from composabl import Agent, Runtime, Skill, SkillCoordinated
from sensors import sensors
from teacher import WalkerTeacher0, WalkerTeacher1, WalkerTeacher2
from coach import CoordinatedCoach

license_key = os.environ["COMPOSABL_LICENSE"]



async def start():

    config = {
        "license": license_key,
        "target": {
            "docker": {
                "image": "composabl/sim-multi-walker"
            },
        },
        "env": {
            "name": "sim-multi-walker",
        },
        "training": {
            "train_batch_size": 5000,
            "replay_buffer_size": 50000,
        },
        "runtime": {
            "num_gpus": 1,
            "workers": 4,
            "envs_per_worker": 4

        }
    }
    runtime = Runtime(config)
    agent = Agent()
    agent.add_sensors(sensors)

    walker_1 = Skill("walker_0", WalkerTeacher0)
    walker_2 = Skill("walker_1", WalkerTeacher1)
    walker_3 = Skill("walker_2", WalkerTeacher2)

    coordinated_skill = SkillCoordinated("coordinated_skill", CoordinatedCoach,
       [walker_1, walker_2, walker_3]
    )

    agent.add_coordinated_skill(coordinated_skill)

    await runtime.train(agent, train_iters=2000)

    agent.export("model")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start())
