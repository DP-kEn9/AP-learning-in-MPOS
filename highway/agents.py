# -*- coding: utf-8 -*-
import torch
import numpy as np
import copy
from policy.q_decom import Q_Decom


class Agents:
    def __init__(self, args, itr=1, agent_id=None, vic=False):
        self.args = copy.deepcopy(args)
        self.vic = vic
        if self.args.training == "adv" and vic:
            self.args.state_shape = int((self.args.state_shape / self.args.n_agents) * \
                                        (self.args.control_units - self.args.n_agents))
            self.args.n_agents = self.args.control_units - self.args.n_agents
            self.args.load_model = True
        self.agent_id = agent_id
        q_decom_policy = ['qmix', 'vdn', 'cwqmix', 'owqmix', 'qatten']
        if args.alg in q_decom_policy:
            self.policy = Q_Decom(self.args, itr)
        else:
            raise Exception("alg unexist")

    def choose_action(self, obs, last_action, agent_idx, avail_actions_mask, epsilon, evaluate=False):
        avail_actions = np.nonzero(avail_actions_mask)[0]
        if np.random.uniform() < epsilon and not evaluate:
            return np.random.choice(avail_actions), None
        if self.args.training == "adv" and not self.vic:
            agent_idx = agent_idx - (self.args.control_units - self.args.n_agents)
        onehot_agent_idx = np.zeros(self.args.n_agents)
        onehot_agent_idx[agent_idx] = 1.
        if self.args.last_action:
            obs = np.hstack((obs, last_action))
        if self.args.reuse_network:
            obs = np.hstack((obs, onehot_agent_idx))
        hidden_state = self.policy.eval_hidden[:, agent_idx, :]
        obs = torch.Tensor(obs).unsqueeze(0)
        avail_actions_mask = torch.Tensor(avail_actions_mask).unsqueeze(0)
        if self.args.cuda:
            obs = obs.cuda()
            hidden_state = hidden_state.cuda()
        with torch.no_grad():
            qsa, self.policy.eval_hidden[:, agent_idx, :] = self.policy.eval_rnn(obs, hidden_state)
        qsa[avail_actions_mask == 0.0] = -float("inf")
        return torch.argmax(qsa), qsa

    def get_max_episode_len(self, batch):
        max_len = 0
        for episode in batch['padded']:
            length = episode.shape[0] - int(episode.sum())
            if length > max_len:
                max_len = length
        return int(max_len)

    def train(self, batch, train_step):
        max_episode_len = self.get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step)
        if train_step > 0 and train_step % self.args.save_model_period == 0:
            self.policy.save_model(train_step)
