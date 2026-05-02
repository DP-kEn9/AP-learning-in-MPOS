

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
from collections import Counter

import copy


class EnvGoOrderly(object):
    def __init__(self, map_size, num_agent):
        self.map_size = map_size
        self.num_agent = num_agent
        self.state = []
        self.goal  = []
        self.order = [[0]*num_agent for _ in range(num_agent)]
        self.former_states = []
        self.occupancy = np.zeros((self.map_size, self.map_size))
        self._episode_steps = 0
        self._episode_limit = map_size*map_size*2








    def reset(self):
        self._episode_steps = 0
        self.state = []
        self.goal = []

        self.former_states = []
        self.occupancy = np.zeros((self.map_size, self.map_size))




























        for i in range(self.num_agent):

            state = [random.randint(1, self.map_size-2), random.randint(1, self.map_size-2)]
            while state in self.state or state in self.goal:
                state = [random.randint(1, self.map_size - 2), random.randint(1, self.map_size - 2)]
            self.state.append(state)
            goal  = [random.randint(1, self.map_size-2), random.randint(1, self.map_size-2)]
            while goal in self.state or goal in self.goal:
                goal = [random.randint(1, self.map_size-2), random.randint(1, self.map_size-2)]
            self.goal.append(goal)












        for i in range(self.map_size):
            self.occupancy[0][i] = 1
            self.occupancy[self.map_size - 1][i] = 1
            self.occupancy[i][0] = 1
            self.occupancy[i][self.map_size - 1] = 1



















    def get_env_info(self):
        env_info = {"state_shape": (4 + self.num_agent) * self.num_agent,
                    "obs_shape": 4 + self.num_agent,
                    "n_actions": 5,
                    "n_agents": self.num_agent,
                    "episode_limit": self._episode_limit,
                    "unit_dim": 2}
        return env_info

    def get_state0(self):
        state = np.zeros((self.num_agent, 4))
        for i in range(self.num_agent):
            state[i,0] = self.state[i][0] / self.map_size
            state[i,1] = self.state[i][1] / self.map_size
            state[i,2] = self.goal[i][0] / self.map_size
            state[i,3] = self.goal[i][1] / self.map_size
        return state

    def get_state(self):
        state = np.zeros((1, (4+self.num_agent)*self.num_agent))
        for i in range(self.num_agent):
            for j in range(4+self.num_agent):
                if j == 0:
                    state[0,(4+self.num_agent)*i+0] = self.state[i][0] / (self.map_size-2)
                elif j == 1:
                    state[0,(4+self.num_agent)*i+1] = self.state[i][1] / (self.map_size-2)
                elif j == 2:
                    state[0,(4+self.num_agent)*i+2] = self.goal[i][0] / (self.map_size-2)
                elif j == 3:
                    state[0,(4+self.num_agent)*i+3] = self.goal[i][1] / (self.map_size-2)
                else:
                    state[0,(4+self.num_agent)*i+j] = self.order[i][j-4]

        return state[0]

    def get_obs(self):
        obs = np.zeros((self.num_agent, 4+self.num_agent))
        for i in range(self.num_agent):
            obs[i, 0] = self.state[i][0] / (self.map_size-2)
            obs[i, 1] = self.state[i][1] / (self.map_size-2)
            obs[i, 2] = self.goal[i][0] / (self.map_size-2)
            obs[i, 3] = self.goal[i][1] / (self.map_size-2)
            for j in range(self.num_agent):
                obs[i, 4+j] = self.order[i][j]
        return obs

    def get_reward(self, state, former_states):

        reward = -1
        reward_flag = True
        num = 0
        for i in range(self.num_agent):
            if state[i] == self.goal[i]:
                num += 1
        if num == self.num_agent:
            return 0
        elif num > 0:
            return float(reward)/num

        return reward

    def get_reward_old(self, state, former_states):
        reward = -1

        reward_flag = True
        num = 0
        for i in range(self.num_agent):
            if state[i] == self.goal[i]:
                num += 1
        if num != self.num_agent:
            reward_flag = False
        else:

            for i in range(self.num_agent):
                idx = self.order[i].index(1)
                consist = len(former_states) - (idx+1)

                for j in range(consist):
                    if former_states[j][i] != self.goal[i]:
                        reward_flag = False
                if consist == 0:
                    if former_states[0][i] == self.goal[i]:
                        reward_flag = False

        if reward_flag:
            reward = 0

        return reward

    def step(self, action_list):
        state = copy.deepcopy(self.state)


        for i in range(self.num_agent):

            if action_list[i] == 0:
                if self.occupancy[self.state[i][0] - 1][self.state[i][1]] != 1:
                    self.state[i][0] = self.state[i][0] - 1
            elif action_list[i] == 1:
                if self.occupancy[self.state[i][0] + 1][self.state[i][1]] != 1:
                    self.state[i][0] = self.state[i][0] + 1
            elif action_list[i] == 2:
                if self.occupancy[self.state[i][0]][self.state[i][1] - 1] != 1:
                    self.state[i][1] = self.state[i][1] - 1
            elif action_list[i] == 3:
                if self.occupancy[self.state[i][0]][self.state[i][1] + 1] != 1:
                    self.state[i][1] = self.state[i][1] + 1
            elif action_list[i] == 4:
                pass

        if len(self.former_states) > 0:
            for i in range(len(self.former_states)-1,0,-1):
                self.former_states[i] = copy.deepcopy(self.former_states[i-1])
            self.former_states[0] = copy.deepcopy(state)






        reward = self.get_reward(self.state, self.former_states)


        self._episode_steps += 1
        info = {'battle_won': False}
        done = False
        if reward >= 0:
            reward = 0
            done = True
            info['battle_won'] = True
        elif self._episode_steps >= self._episode_limit:
            done = True
        return reward, done, info

    def sqr_dist(self, pos1, pos2):
        return (pos1[0]-pos2[0])*(pos1[0]-pos2[0])+(pos1[1]-pos2[1])*(pos1[1]-pos2[1])

    def get_global_obs(self):
        obs = np.zeros((self.map_size, self.map_size, 4))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.occupancy[i][j] == 0:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
                    obs[i, j, 3] = 1.0
        for i in range(self.num_agent):
            if i%6 == 0:

                obs[self.state[i][0], self.state[i][1], 0] = 1.0
                obs[self.state[i][0], self.state[i][1], 1] = 0.0
                obs[self.state[i][0], self.state[i][1], 2] = 0.0
                obs[self.state[i][0], self.state[i][1], 3] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 0.0
            elif i%6 == 1:

                obs[self.state[i][0], self.state[i][1], 0] = 0.0
                obs[self.state[i][0], self.state[i][1], 1] = 1.0
                obs[self.state[i][0], self.state[i][1], 2] = 0.0
                obs[self.state[i][0], self.state[i][1], 3] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 0.0
            elif i%6 == 2:

                obs[self.state[i][0], self.state[i][1], 0] = 0.0
                obs[self.state[i][0], self.state[i][1], 1] = 0.0
                obs[self.state[i][0], self.state[i][1], 2] = 1.0
                obs[self.state[i][0], self.state[i][1], 3] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 1.0
            elif i%6 == 3:

                obs[self.state[i][0], self.state[i][1], 0] = 0.0
                obs[self.state[i][0], self.state[i][1], 1] = 0.0
                obs[self.state[i][0], self.state[i][1], 2] = 0.0
                obs[self.state[i][0], self.state[i][1], 3] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 1.0
            elif i%6 == 4:

                obs[self.state[i][0], self.state[i][1], 0] = 1.0
                obs[self.state[i][0], self.state[i][1], 1] = 0.0
                obs[self.state[i][0], self.state[i][1], 2] = 1.0
                obs[self.state[i][0], self.state[i][1], 3] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 0.0
            else:

                obs[self.state[i][0], self.state[i][1], 0] = 0.0
                obs[self.state[i][0], self.state[i][1], 1] = 1.0
                obs[self.state[i][0], self.state[i][1], 2] = 0.0
                obs[self.state[i][0], self.state[i][1], 3] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 1.0

        return obs

    def plot_scene(self):
        plt.figure(figsize=(5, 5))
        plt.imshow(self.get_global_obs())
        plt.xticks([])
        plt.yticks([])
        plt.show()

























































    def get_avail_agent_actions(self, agent_id):
        return [1] * 5

    def close(self):
        pass
