from composabl import SkillTeacher

from sensors import get_sensor_names

class WalkerTeacher0(SkillTeacher):
    def __init__(self):
        self.past_obs = None

    async def compute_reward(self, transformed_obs, action, sim_reward):
        return sim_reward

    async def compute_action_mask(self, transformed_obs, action):
        return None

    async def compute_success_criteria(self, transformed_obs, action):
        return False

    async def compute_termination(self, transformed_obs, action):
        return False

    async def transform_obs(self, obs, action):
        return obs

    async def transform_action(self, transformed_obs, action):
        return action

    async def filtered_observation_space(self):
        return get_sensor_names(0)

class WalkerTeacher1(SkillTeacher):
    def __init__(self):
        self.past_obs = None

    async def compute_reward(self, transformed_obs, action, sim_reward):
        return sim_reward

    async def compute_action_mask(self, transformed_obs, action):
        return None

    async def compute_success_criteria(self, transformed_obs, action):
        return False

    async def compute_termination(self, transformed_obs, action):
        return False

    async def transform_obs(self, obs, action):
        return obs

    async def transform_action(self, transformed_obs, action):
        return action

    async def filtered_observation_space(self):
        return get_sensor_names(1)

class WalkerTeacher2(SkillTeacher):
    def __init__(self):
        self.past_obs = None

    async def compute_reward(self, transformed_obs, action, sim_reward):
        return sim_reward

    async def compute_action_mask(self, transformed_obs, action):
        return None

    async def compute_success_criteria(self, transformed_obs, action):
        return False

    async def compute_termination(self, transformed_obs, action):
        return False

    async def transform_obs(self, obs, action):
        return obs

    async def transform_action(self, transformed_obs, action):
        return action

    async def filtered_observation_space(self):
        return get_sensor_names(2)
