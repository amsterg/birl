from env import SimpleEnv
import numpy as np
from gym.envs.toy_text import FrozenLakeEnv
np.random.seed(42)


class DP:
    def __init__(self, env, gamma=1e-1):
        self.state_values = np.round(
            (np.random.uniform(size=[env.num_states])*0.001), 0)
        self.env = env
        self.gamma = gamma
        self.policy = np.ones(
            [self.env.num_states, self.env.num_actions])/self.env.num_actions
        # print(self.policy)
        # print(self.state_values)

    def policy_eval(self):
        print(self.state_values)
        state_values_new = []
        delta = 0
        for s, sv in enumerate(self.state_values):
            # vs = sv
            # for a,ap in enumerate(self.policy[s]):
            #     vs += ap*self.env.s_t_probs[s][a]*(self.env.rewards[s] + self.gamma*self.state_values[s])
            sv += np.dot(np.reshape(self.policy[s], (1, -1)), (self.env.s_t_probs[s]))*(
                self.env.rewards[s] + self.gamma*self.state_values[s])
            state_values_new = sv

        self.state_values = np.ndarray.flatten(state_values_new)

    def policy_imp(self):
        actions = []
        print(self.state_values)
        # for a in self.env.num_actions:
        #     for s, sv in enumerate(self.state_values):
        #         a = self.env.s_t_probs[s][a] * (self.env.rewards[s] +
        #                                         self.gamma*self.state_values[s])
        for s, sv in enumerate(self.state_values):
            action_vals = []
            for a in range(self.env.num_actions):
                action_vals.append(np.multiply(self.env.s_t_probs[s][a], (self.env.rewards[s] +
                                                                          self.gamma*self.state_values[s])).sum())

                # action_vals = np.dot(self.env.s_t_probs[s], (self.env.rewards[s] +
                # self.gamma*self.state_values[s]))
            action_new = np.argmax(action_vals)
            self.policy[s] = action_vals

            actions.append(action_new)
        print(actions)


if __name__ == "__main__":
    env = SimpleEnv()
    dp = DP(env)
    while True:
        dp.policy_eval()
        dp.policy_imp()
        break
    # env=FrozenLakeEnv()
    # env.num_actions=env.nA
    # env.num_states=env.nS
    # env.reset()
    # for _ in range(1000):
    #     env.render()
    #     env.step(env.action_space.sample())  # take a random action
    # env.close()
    # dp=DP(env)
    # print(env.__dict__)
    # dp.policy_eval()
