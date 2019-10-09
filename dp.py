from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
from env import SimpleEnv, FrozenEnv, FrozenLakeEnv
import numpy as np
# from gym.envs.toy_text import FrozenLakeEnv
np.random.seed(42)


class DP:
    def __init__(self, env, gamma=0.8):
        self.env = env
        self.gamma = gamma

        self.state_values = np.ones(
            self.env.num_states)*100/self.env.num_states
        self.policy = np.zeros(
            [self.env.num_states, self.env.num_actions])/self.env.num_actions
        self.q_values = {s: np.ones(self.env.num_actions)
                         for s in range(self.env.num_states)}

    def policy_eval(self):
        while True:
            delta = 0.0
            delta_thres = 1e-5

            for s in range(self.env.num_states):
                sv = 0
                for a, ap in enumerate(self.policy[s]):
                    for p, ns, r, d in self.env.P[s][a]:
                        sv += ap * p * (r + self.gamma * self.state_values[ns])
                # print(sv)
                # np.dot(
                #     self.policy[s],np.multiply(
                #         self.env.P[s]
                #     )
                # )
                # exit()
                delta = max(delta, np.abs(sv-self.state_values[s]))

                self.state_values[s] = sv
            # print("delta: ",delta)
            if delta < delta_thres:
                break
            # break

    def policy_imp(self):
        policy_stable = True
        for s in range(self.env.num_states):
            curr_action = np.argmax(self.policy[s])
            action_vals = np.zeros(self.env.num_actions)
            for a in range(self.env.num_actions):
                for p, ns, r, d in self.env.P[s][a]:
                    av = p * (r + self.gamma * self.state_values[ns])
                    action_vals[a] += av
                # print(s)
            self.q_values[s] = action_vals
            action_best = np.argmax(action_vals)
            if action_best != curr_action:
                policy_stable = False
            self.policy[s] = np.eye(self.env.num_actions)[action_best]

    def policy_iter(self):
        while True:
            old_policy = deepcopy(self.policy)
            self.policy_eval()
            self.policy_imp()
            if np.all(old_policy == self.policy):
                break


if __name__ == "__main__":

    env = FrozenLakeEnv(is_slippery=True)

    env.num_actions = env.nA
    env.num_states = env.nS
    o = env.reset()
    dp = DP(env)
    i = 0
    # for s in env.P:
    #     for a in env.P[s]:
    #         i += 1
    #         print(env.P[s][a])
    # print(i)
    # print(env.P)
    for _ in range(200):
        dp.policy_eval()
        dp.policy_imp()
    plt.figure(figsize=(8, 8))
    sns.heatmap(dp.state_values.reshape(4, 4),
                cmap="YlGnBu", annot=True, cbar=False)
    plt.show()

    done = False
    while True:
        env.render()
        o, _, done, _ = env.step(np.argmax(dp.policy[o]))
        if done:
            if o == env.num_states-1:
                env.render()

                break
            else:
                env.reset()

    env.close()
