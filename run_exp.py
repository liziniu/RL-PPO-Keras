from env import Env
from ppo import Agent


def main(dic_agent_conf, dic_env_conf, dic_exp_conf, dic_path):
    env = Env(dic_env_conf)

    dic_agent_conf["ACTION_DIM"] = env.action_dim
    dic_agent_conf["STATE_DIM"] = (env.state_dim, )

    agent = Agent(dic_agent_conf, dic_path, dic_env_conf)

    for cnt_episode in range(dic_exp_conf["TRAIN_ITERATIONS"]):
        s = env.reset()
        r_sum = 0
        for cnt_step in range(dic_exp_conf["MAX_EPISODE_LENGTH"]):
            if cnt_episode > dic_exp_conf["TRAIN_ITERATIONS"] - 10:
                env.render()

            a = agent.choose_action(s)
            s_, r, done, _ = env.step(a)

            r /= 100
            r_sum += r
            if done:
                r = -1

            agent.store_transition(s, a, s_, r, done)
            if cnt_step % dic_agent_conf["BATCH_SIZE"] == 0 and cnt_step != 0:
                agent.train_network()
            s = s_

            if done:
                break

            if cnt_step % 10 == 0:
                print("Episode:{}, step:{}, r_sum:{}".format(cnt_episode, cnt_step, r_sum))


