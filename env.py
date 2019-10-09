import sys
from gym.envs.toy_text import discrete
from gym import utils
from six import StringIO, b
from contextlib import closing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from gym.envs.toy_text import FrozenLakeEnv


class SimpleEnv:
    def __init__(self, num_states=4, num_actions=2, rewards=[0, 0, 0, 1]):
        self.num_actions = num_actions
        self.num_states = num_states
        self.states = np.array(range(self.num_states))
        np.random.seed(42)
        self.P = np.random.uniform(low=0, high=0.25, size=(
            self.num_states, self.num_actions, self.num_states))
        self.P = self.P / \
            np.sum(self.P, axis=2, keepdims=1)
        # print(np.sum(self.P,axis = 2))
        # print(self.P)
        self.state = None
        self.rewards = rewards
        pass

    def step(self, action):
        self.state = np.random.choice(
            self.states, p=self.P[self.state, action])
        return self.state, self.rewards[self.state]

    def reset(self):
        self.state = 0
        return self.state


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if (res[r_new][c_new] not in '#H'):
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]


class FrozenLakeEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4", is_slippery=True, rewards=None):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)
        if rewards is not None:
            self.rewards = rewards
            assert len(self.rewards) == self.nrow*self.ncol
        else:
            self.rewards = None

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col-1, 0)
            elif a == DOWN:
                row = min(row+1, nrow-1)
            elif a == RIGHT:
                col = min(col+1, ncol-1)
            elif a == UP:
                row = max(row-1, 0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a-1) % 4, a, (a+1) % 4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                rew = float(newletter == b'G')
                                if self.rewards is not None:
                                    rew = self.rewards[s]
                                li.append((1.0/3.0, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            rew = float(newletter == b'G')
                            li.append((1.0, newstate, rew, done))

        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(
                ["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()


class FrozenEnv(FrozenLakeEnv):
    def __init__(self):
        super().__init__()
        self.num_actions = self.nA
        self.num_states = self.nS


if __name__ == "__main__":
    # env = SimpleEnv()
    # env.reset()
    # plt.figure(figsize=(8, 8))
    # print(env.P)
    # sns.heatmap(env.P,
    #             cmap="YlGnBu", annot=True, cbar=False)
    # plt.show()

    env = FrozenEnv()
    # print(env.num_actions)
    # print(env.num_states)

    # o = env.reset()
    # dp = DP(env)

    # for _ in range(2000):
    #     dp.policy_eval()
    #     dp.policy_imp()
    # plt.figure(figsize=(8, 8))
    # sns.heatmap(dp.state_values.reshape(4, 4),
    #             cmap="YlGnBu", annot=True, cbar=False)
    # plt.show()

    # for _ in range(2000):
    #     env.render()
    #     o, _, _, _ = env.step(np.argmax(dp.policy[o]))  # take a random action
