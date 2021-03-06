import copy
from dp import DP
import matplotlib.pyplot as plt
import seaborn as sns
from env import FrozenLakeEnv
import numpy as np
np.random.seed(42)
import os
from time import sleep

class Birl():
    def __init__(self, num_states):
        self.num_states = num_states
        self.gamma = 0.8
        self.alpha = 10
        self.sim_store = None
        self.rewards_recovered = None

    def sample_random_rewards(self, n_states, step_size, r_max):
        """
        sample random rewards form gridpoint(R^{n_states}/step_size).
        :param n_states:
        :param step_size:
        :param r_max:
        :return: sampled rewards
        """
        rewards = np.random.uniform(low=-r_max, high=r_max, size=n_states)

        # move these random rewards toward a gridpoint
        # add r_max to makee mod to be always positive
        # add step_size for easier clipping
        rewards = rewards + r_max + step_size

        for i, reward in enumerate(rewards):
            mod = reward % step_size
            rewards[i] = reward - mod
        # subtracts added values from rewards
        rewards = rewards - (r_max + step_size)
        return rewards

    def mcmc_reward_step(self, rewards, step_size, r_max):
        new_rewards = copy.deepcopy(rewards)
        index = np.random.randint(len(rewards))
        step = np.random.choice([-step_size, step_size])
        new_rewards[index] += step
        new_rewards = np.clip(a=new_rewards, a_min=-r_max, a_max=r_max)
        if np.all(new_rewards == rewards):
            new_rewards[index] -= step
        assert np.any(rewards != new_rewards), 'rewards do not change: {}, {}'.format(
            new_rewards, rewards)
        return new_rewards

    def optimal_q_check(self, q_values, pi):
        assert q_values.shape == pi.shape, "Shapes mismatch for qvalues in qs_comp"
        for s in range(q_values.shape[0]):
            for a in range(q_values.shape[1]):
                if q_values[s, a] > q_values[s, np.argmax(pi[s])]:
                    # if atleast one (s,a) exists that is to be optimizied(kinda)
                    return True
        return False

    def posterior(self, agent_with_env, prior):
        agent_with_env.policy_imp()
        q_vals = agent_with_env.q_values
        return np.sum([self.alpha * q_vals[s][a] - np.log(np.sum(np.exp(self.alpha * q_vals[s]))) for s, a in self.sim_store]) + np.log(prior)

    def posteriors_ratio(self, dp, dp_new, prior=1):
        ln_p_new = self.posterior(dp_new, prior)
        ln_p = self.posterior(dp, prior)
        return np.exp(
            ln_p_new - ln_p
        )

    def policy_walk(self):
        random_rewards = self.sample_random_rewards(self.num_states, 1, 1)
        env = FrozenLakeEnv(is_slippery=True, rewards=random_rewards)
        env.num_actions = env.nA
        env.num_states = env.nS
        o = env.reset()
        dp = DP(env)
        # for _ in range(1):
        #     dp.policy_eval()
        #     dp.policy_imp()
        dp.policy_iter()

        dp.q_values = np.array([dp.q_values[s] for s in dp.q_values])
        pi = dp.policy
        # plt.figure(figsize=(8, 8),num="pi")
        # sns.heatmap(dp.policy.reshape(16, 4),
        #             cmap="Spectral", annot=True, cbar=False)
        # plt.show()
        for _ in range(200):
            random_rewards = env.rewards
            new_rewards = self.mcmc_reward_step(
                random_rewards, step_size=0.5, r_max=1)
            # new_rewards = self.sample_random_rewards(self.num_states, 1, 1)
            env_new = FrozenLakeEnv(is_slippery=True, rewards=new_rewards)
            env_new.num_actions = env_new.nA
            env_new.num_states = env_new.nS
            # o = env_new.reset()
            dp_new = DP(env_new)
            # print(dp_new.policy)
            # dp_new = DP(env_new)
            # plt.figure(figsize=(8, 8),num="pi before imp")
            # sns.heatmap(dp.policy.reshape(16, 4),
            #         cmap="Spectral", annot=True, cbar=False)

            dp_new.policy_iter()

            # dp_new.policy_imp()
            dp_new_q_values = np.array([dp_new.q_values[s]
                                        for s in dp_new.q_values])
            dp_new = DP(env_new)
            dp_new.policy = pi
            dp_new.q_values = dp_new_q_values

            # plt.figure(figsize=(8, 8),num="pi after imp")
            # sns.heatmap(dp.policy.reshape(16, 4),
            #         cmap="Spectral", annot=True, cbar=False)
            # plt.figure(figsize=(8, 8),num="new q's")

            # sns.heatmap(dp_new.q_values.reshape(16, 4),
            #         cmap="Spectral", annot=True, cbar=False)
            # plt.show()

            """
            if "dp_q_values < dp_new_q_values":
                    or
            if "dp_new_q_values(pi) < dp_new_q_values" (with this for now):
            
            """
            if self.optimal_q_check(dp_new.q_values, pi):
                dp_new.policy_iter()
                pi_new = dp_new.policy
                """
                prob_comparision = update env(rews) policy with prob ( min(1, ratio(posterioirs of dp,dp_new's policies)))
                """
                # if posteriors_ratio(env_new,pi_new,env,pi,prior,)
                if np.random.random() < self.posteriors_ratio(dp, dp_new):
                    print("update env and pi")

                    # "porb comparision":
                    env, pi = env_new, pi_new
            else:
                if np.random.random() < self.posteriors_ratio(dp, dp_new):
                    # if "prob comparision":
                    print("update env")

                    env = env_new

            # break
        self.rewards_recovered = env.rewards

        # plt.figure(figsize=(8, 8), num="v of dp")
        # sns.heatmap(dp.state_values.reshape(4, 4),
        #             cmap="Spectral", annot=True, cbar=False)
        # plt.figure(figsize=(8, 8), num="v of dp_new")
        # sns.heatmap(dp_new.state_values.reshape(4, 4),
        #             cmap="Spectral", annot=True, cbar=False)
        # plt.figure(figsize=(8, 8))
        # sns.heatmap(dp.q_values.reshape(16, 4),
        #             cmap="Spectral", annot=True, cbar=False)
        # plt.show()

    def sim(self, agent_with_env):
        done = False
        sim_store = []
        env = agent_with_env.env
        policy = agent_with_env.policy
        o = agent_with_env.env.reset()
        ix = 0
        while True:
            ix+=1
            env.render()
            action = np.argmax(policy[o])
            sim_store.append([o, action])
            o, _, done, _ = env.step(action)
            if done:
                if o == env.num_states-1:
                    env.render()

                    break
                else:
                    env.reset()
        print(ix)
        sleep(1)
        env.close()
        return sim_store


if __name__ == "__main__":
    env = FrozenLakeEnv(is_slippery=True)

    env.num_actions = env.nA
    env.num_states = env.nS
    o = env.reset()
    dp = DP(env)
    for _ in range(100):
        dp.policy_eval()
        dp.policy_imp()
    dp.q_values = np.array([dp.q_values[s] for s in dp.q_values])
    # exit()
    # plt.figure(figsize=(8, 8), num="dp_sv")
    # sns.heatmap(dp.state_values.reshape(4, 4),
    #             cmap="Spectral", annot=True, cbar=False)
    # plt.figure(figsize=(8, 8), num="dp_q")
    # sns.heatmap(dp.q_values.reshape(16, 4),
    #             cmap="Spectral", annot=True, cbar=False)
    # plt.show()
    # plt.show()
    birl = Birl(env.num_states)
    print("Running Sim")
    birl.sim_store = birl.sim(dp)
    print("Running Sim Done")
    birl.policy_walk()
    rewards_implicit = np.array([sum([val[2] for a in env.P[s] for val in env.P[s][a]])
                        for s in env.P])

    plt.figure(figsize=(8, 8), num="reward_original")
    sns.heatmap(rewards_implicit.reshape(4, 4),
                cmap="Spectral", annot=True, cbar=False)
    plt.figure(figsize=(8, 8), num="reward_recovered")
    sns.heatmap(birl.rewards_recovered.reshape(4, 4),
                cmap="Spectral", annot=True, cbar=False)
    plt.show()

    env_gen_rews = FrozenLakeEnv(is_slippery=True,rewards=birl.rewards_recovered)

    env_gen_rews.num_actions = env_gen_rews.nA
    env_gen_rews.num_states = env_gen_rews.nS
    o = env_gen_rews.reset()
    dp_rg = DP(env_gen_rews)
    os.system('clear')
    sleep(1)
    
    for _ in range(100):
        dp_rg.policy_eval()
        dp_rg.policy_imp()
    plt.figure(figsize=(8, 8), num="dp_sv")
    sns.heatmap(dp.state_values.reshape(4, 4)/(np.max(dp.state_values)-np.min(dp.state_values)),
                cmap="Spectral", annot=True, cbar=False)
    plt.figure(figsize=(8, 8), num="gen_r_dp_sv")
    sns.heatmap(dp_rg.state_values.reshape(4, 4)/np.max(dp_rg.state_values)-np.min(dp_rg.state_values),
                cmap="Spectral", annot=True, cbar=False)
    plt.show()
    birl_new = Birl(env_gen_rews.num_states)
    birl_new.sim(dp_rg)

    
    # print(birl.policy)
    # print(dp.q_values.shape)

    # done = False
    # while True:
    #     env.render()
    #     o, _, done, _ = env.step(np.argmax(dp.policy[o]))
    #     if done:
    #         if o == env.num_states-1:
    #             env.render()

    #             break
    #         else:
    #             env.reset()

    # env.close()
