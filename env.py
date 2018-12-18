import gym


class Env:
    def __init__(self, dic_env_config):
        self.dic_env_config = dic_env_config

        self.env = gym.make(self.dic_env_config["ENV_NAME"])
        self.env.seed(self.dic_env_config["GYM_SEED"])

    def reset(self):
        return self.env.reset()

    def step(self, a):
        return self.env.step(a)

    def render(self):
        self.env.render()

    @property
    def state_dim(self):
        return self.env.observation_space.shape[0]

    @property
    def action_dim(self):
        return self.env.action_space.n