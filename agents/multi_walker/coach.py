from composabl_core.agent.skill.skill_coach import SkillCoach

class CoordinatedCoach(SkillCoach):
    def __init__(self):
        self.counter = 0
        self.agent_pos = {
            "walker_1": [0, 0],
            "walker_2": [0, 0],
            "walker_3": [0, 0]
        }

    async def compute_reward(self, transformed_obs, action, sim_reward):
        # just use the sim_reward for now
        total_reward = 0
        for skill_name in sim_reward:
            total_reward += sim_reward[skill_name]
        return total_reward / len(sim_reward)

    async def compute_action_mask(self, transformed_obs, action):
        return None

    async def compute_success_criteria(self, transformed_obs, action):
        return False

    async def compute_termination(self, transformed_obs, action):
        return False

    async def transform_action(self, transformed_obs, action):
        return action
