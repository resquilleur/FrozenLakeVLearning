import gym
from tensorboardX import SummaryWriter
import Agent

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_env = gym.make(ENV_NAME, map_name="4x4", is_slippery=False)
    agent = Agent.Agent(env_name=ENV_NAME, gamma=GAMMA)
    writer = SummaryWriter(comment="-v-iteration")
    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward update %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()


