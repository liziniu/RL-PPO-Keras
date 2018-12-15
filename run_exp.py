import numpy as np
import keras
from env import Env
from config import dic_env_conf, dic_agent_conf, dic_path, dic_exp_conf
from ppo import Agent

env = Env(dic_env_conf)
agent = Agent(dic_agent_conf, dic_path, dic_env_conf)


for cnt_episode in range(dic_env_conf["NUM_EPISODE"]):
    s = env.reset()

    for i in range(dic_env_conf["MAX_EPISODE_LENGTH"]):
        a = agent.choose_action(s)
        s_, r, done, _ = env.step(a)

        agent.store_transition(s, a, s_, r)
        if i % dic_env_conf["BATCH_SIZE"] == 0:
            agent.train_network()

        s = s_

