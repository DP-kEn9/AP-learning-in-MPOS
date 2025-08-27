# -*- coding: utf-8 -*-
import numpy as np
import threading
from torch import nn


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.current_size = 0
        # self.buffer = {'o': collections.deque(maxlen=self.args.buffer_size),
        #                's': collections.deque(maxlen=self.args.buffer_size),
        #                'a': collections.deque(maxlen=self.args.buffer_size),
        #                'onehot_a': collections.deque(maxlen=self.args.buffer_size),
        #                'avail_a': collections.deque(maxlen=self.args.buffer_size),
        #                'r': collections.deque(maxlen=self.args.buffer_size),
        #                'next_o': collections.deque(maxlen=self.args.buffer_size),
        #                'next_s': collections.deque(maxlen=self.args.buffer_size),
        #                'next_avail_a': collections.deque(maxlen=self.args.buffer_size),
        #                'done': collections.deque(maxlen=self.args.buffer_size),
        #                'padded': collections.deque(maxlen=self.args.buffer_size)
        #                }
        self.buffer = {
            'o': np.zeros(
                [self.args.buffer_size, self.args.episode_limit, self.args.n_agents, self.args.obs_shape]),
            's': np.zeros(
                [self.args.buffer_size, self.args.episode_limit, self.args.state_shape]),
            'a': np.zeros(
                [self.args.buffer_size, self.args.episode_limit, self.args.n_agents, 1]),
            'onehot_a': np.zeros(
                [self.args.buffer_size, self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
            'avail_a': np.zeros(
                [self.args.buffer_size, self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
            'r': np.zeros(
                [self.args.buffer_size, self.args.episode_limit, 1]),
            'r_m': np.zeros(
                [self.args.buffer_size, self.args.episode_limit, 1]),
            'next_o': np.zeros(
                [self.args.buffer_size, self.args.episode_limit, self.args.n_agents, self.args.obs_shape]),
            'next_s': np.zeros(
                [self.args.buffer_size, self.args.episode_limit, self.args.state_shape]),
            'next_avail_a': np.zeros(
                [self.args.buffer_size, self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
            'done': np.ones(
                [self.args.buffer_size, self.args.episode_limit, 1]),
            'padded': np.ones(
                [self.args.buffer_size, self.args.episode_limit, 1]),
            'ap': np.zeros([self.args.buffer_size, self.args.episode_limit, 1])
        }
        self.current_idx = 0
        self.size = 0
        self.lock = threading.Lock()

    def sample(self, batch_size):
        temp_buffer = {}
        idxes = np.random.randint(0, self.size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idxes]
        return temp_buffer

    def store(self, episode_batch):
        with self.lock:
            num = episode_batch['o'].shape[0]
            idxes = self.get_idxes(num)
            for key in self.buffer.keys():
                self.buffer[key][idxes] = episode_batch[key]
            self.size = min(self.args.buffer_size, self.size + num)

    def get_idxes(self, num):
        if self.current_idx + num <= self.args.buffer_size:
            idxes = np.arange(self.current_idx, self.current_idx + num)
            self.current_idx += num
        elif self.current_idx < self.args.buffer_size:
            overflow = num - (self.args.buffer_size - self.current_idx)
            idxes = np.concatenate([np.arange(self.current_idx, self.args.buffer_size),
                                    np.arange(0, overflow)])
            self.current_idx = overflow
        else:
            idxes = np.arange(0, num)
            self.current_idx = num
        return idxes

