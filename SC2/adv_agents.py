import torch
import numpy as np
import copy
from policy.adv_q_decom import Q_Decom
from policy.PPO2 import PPO


class AdvAgents:
    def __init__(self, args, itr=1, agent_id=None, load_model=False):
        self.args = copy.copy(args)
        # self.args.n_agents = self.args.n_agents + self.args.n_adv
        self.agent_id = agent_id
        q_decom_policy = ['qmix', 'vdn', 'cwqmix', 'owqmix', 'qatten']
        if args.adv_alg in q_decom_policy:
            self.policy = Q_Decom(self.args, itr, load_model=load_model)
        elif 'PPO' in args.adv_alg:
            self.policy = PPO(args)
        else:
            raise Exception("unexist")

    def choose_action(self, obs, last_action, agent_idx, avail_actions_mask, epsilon, evaluate=False):
        avail_actions_mask = avail_actions_mask[:6]
        avail_actions = np.nonzero(avail_actions_mask)[0]
        if np.random.uniform() < epsilon and not evaluate and 'PPO' not in self.args.adv_alg:
            return np.random.choice(avail_actions), None
        onehot_agent_idx = np.zeros(self.args.n_agents)
        onehot_agent_idx[agent_idx] = 1.
        if self.args.last_action:
            obs = np.hstack((obs, last_action))
        if self.args.reuse_network:
            obs = np.hstack((obs, onehot_agent_idx))

        obs = torch.Tensor(obs).unsqueeze(0)
        avail_actions_mask = torch.Tensor(avail_actions_mask).unsqueeze(0)
        if 'PPO' in self.args.adv_alg:
            if self.args.cuda:
                obs = obs.cuda()
            return self.policy.choose_action(obs, avail_actions)
        hidden_state = self.policy.eval_hidden[:, agent_idx - (self.args.n_agents - self.args.n_adv), :]
        if self.args.cuda:
            obs = obs.cuda()
            hidden_state = hidden_state.cuda()
        with torch.no_grad():
            qsa, self.policy.eval_hidden[:, agent_idx - (self.args.n_agents - self.args.n_adv), :] = \
                self.policy.eval_rnn(obs, hidden_state)
        # qsa, self.policy.eval_hidden[:, agent_idx - (self.args.n_agents - self.args.n_adv), :] = \
        #     self.policy.eval_rnn(obs, hidden_state)
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
