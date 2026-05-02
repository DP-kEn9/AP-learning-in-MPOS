

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env.multiagentenv import MultiAgentEnv

import atexit
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
from absl import logging
import random


class uni_mmdp_Env(MultiAgentEnv):

    def __init__(
            self,
            n_agents=2,
            n_actions=1,

            episode_limit=50,
            obs_last_action=True,
            state_last_action=True,
            is_print=False,
            print_rew=False,
            print_steps=1000,
            seed=None
    ):

        self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)
        self.n_agents = n_agents





        self._seed = seed


        self.n_states = episode_limit
        self.n_actions = n_actions


        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0

        self.p_step = 0
        self.rew_gather = []
        self.is_print_once = False

        self.episode_limit = episode_limit

        self.R = np.zeros((self.n_states, self.n_actions, self.n_actions))






        self.T = np.ones((self.n_states, self.n_actions, self.n_actions)).astype('int32')
        for i in range(episode_limit):
            self.T[i] = [[i+1]]

        self.state_now = 0


        self.unit_dim = self.n_states

    def step(self, actions):

        self._total_steps += 1
        self._episode_steps += 1
        info = {}

        reward = self.R[self.state_now][actions[0]][actions[1]]
        self.state_now = self.T[self.state_now][actions[0]][actions[1]]

        terminated = False
        info['battle_won'] = False

        if self._episode_steps >= self.episode_limit-1:
            terminated = True
            self._episode_steps = 0

        if terminated:
            self._episode_count += 1
            self.battles_game += 1

        return reward, terminated, info

    def get_obs(self):

        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):

        return np.eye(self.n_states)[self.state_now]

    def get_obs_size(self):

        return self.n_states

    def get_state(self):

        return np.concatenate([np.eye(self.n_states)[self.state_now] for _ in range(self.n_agents)], axis=0)

    def get_state_size(self):

        return self.n_agents * self.n_states

    def get_avail_actions(self):

        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):

        return [1] * self.n_actions

    def get_total_actions(self):

        return self.n_actions

    def reset(self):

        self._episode_steps = 0
        self.state_now = 1

        return self.get_obs(), self.get_state()

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

    def save_replay(self):

        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit,
                    "unit_dim": self.unit_dim}
        return env_info

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "win_rate": self.battles_won / self.battles_game
        }
        return stats
