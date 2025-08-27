import torch
import os

from torch import nn

from network.base_net import RNN
from network.qmix_mixer import QMIXMixer
from network.vdn_mixer import VDNMixer
from network.wqmix_q_star import QStar
from network.qatten_mixer import QattenMixer
import numpy as np


class Q_Decom:
    def __init__(self, args, itr):
        self.args = args
        input_shape = self.args.obs_shape
        if args.last_action:
            input_shape += self.args.n_actions
        if args.reuse_network:
            input_shape += self.args.n_agents
        self.eval_rnn = RNN(input_shape, args)
        self.target_rnn = RNN(input_shape, args)
        self.wqmix = 0
        if self.args.alg == 'cwqmix' or self.args.alg == 'owqmix':
            self.wqmix = 1

        if 'qmix' in self.args.alg:
            self.eval_mix_net = QMIXMixer(args)
            self.target_mix_net = QMIXMixer(args)
            if self.wqmix > 0:
                self.qstar_eval_mix = QStar(args)
                self.qstar_target_mix = QStar(args)
                self.qstar_eval_rnn = RNN(input_shape, args)
                self.qstar_target_rnn = RNN(input_shape, args)
                if self.args.alg == 'cwqmix':
                    self.alpha = 0.75
                elif self.args.alg == 'owqmix':
                    self.alpha = 0.5
                else:
                    raise Exception('')
        elif self.args.alg == 'vdn':
            self.eval_mix_net = VDNMixer()
            self.target_mix_net = VDNMixer()
        elif self.args.alg == 'qatten':
            self.eval_mix_net = QattenMixer(args)
            self.target_mix_net = QattenMixer(args)

        if args.cuda:
            torch.distributed.init_process_group(backend="nccl")
            self.eval_rnn, self.target_rnn, self.eval_mix_net, self.target_mix_net = \
                self.eval_rnn.to(args.device), self.target_rnn.to(args.device), self.eval_mix_net.to(
                    args.device), self.target_mix_net.to(args.device)
            self.eval_rnn, self.target_rnn, self.eval_mix_net, self.target_mix_net = \
                nn.parallel.DistributedDataParallel(self.eval_rnn, device_ids=[args.local_rank]), \
                nn.parallel.DistributedDataParallel(self.target_rnn, device_ids=[args.local_rank]), \
                nn.parallel.DistributedDataParallel(self.eval_mix_net, device_ids=[args.local_rank]), \
                nn.parallel.DistributedDataParallel(self.target_mix_net, device_ids=[args.local_rank])

            # self.eval_rnn.cuda()
            # self.target_rnn.cuda()
            # self.eval_mix_net.cuda()
            # self.target_mix_net.cuda()
            if self.wqmix > 0:
                self.qstar_eval_mix, self.qstar_target_mix, self.qstar_eval_rnn, self.qstar_target_rnn = \
                    self.qstar_eval_mix.to(args.device), self.qstar_target_mix.to(args.device), self.qstar_eval_rnn.to(
                        args.device), self.qstar_target_rnn.to(args.device)
                self.qstar_eval_mix, self.qstar_target_mix, self.qstar_eval_rnn, self.qstar_target_rnn = \
                    nn.parallel.DistributedDataParallel(self.qstar_eval_mix, device_ids=[args.local_rank]), \
                    nn.parallel.DistributedDataParallel(self.qstar_target_mix, device_ids=[args.local_rank]), \
                    nn.parallel.DistributedDataParallel(self.qstar_eval_rnn, device_ids=[args.local_rank]), \
                    nn.parallel.DistributedDataParallel(self.qstar_target_rnn, device_ids=[args.local_rank])

                # self.qstar_eval_mix.cuda()
                # self.qstar_target_mix.cuda()
                # self.qstar_eval_rnn.cuda()
                # self.qstar_target_rnn.cuda()
        # self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map + '/' + str(itr)
        # self.model_dir = args.model_dir + '/' + args.map + '/' + args.alg + '_' + str(self.args.epsilon_anneal_steps // 10000) + 'w' + '/' + str(itr)
        if args.load_model:
            if os.path.exists(self.args.load_dir + '/rnn_net_params.pkl'):
                path_rnn = self.args.load_dir + '/rnn_net_params.pkl'
                path_mix = self.args.load_dir + '/' + self.args.alg + '_net_params.pkl'

                map_location = 'cuda:0' if args.cuda else 'cpu'

                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_mix_net.load_state_dict(torch.load(path_mix, map_location=map_location))
                if self.wqmix > 0:
                    path_agent_rnn = self.args.load_dir + '/rnn_net_params2.pkl'
                    path_qstar = self.args.load_dir + '/' + 'qstar_net_params.pkl'
                    self.qstar_eval_rnn.load_state_dict(torch.load(path_agent_rnn, map_location=map_location))
                    self.qstar_eval_mix.load_state_dict(torch.load(path_qstar, map_location=map_location))
                print(' %s'%path_rnn + ' 和 %s'%path_mix)
            else:
                raise Exception("")
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
        self.eval_params = list(self.eval_rnn.parameters()) + list(self.eval_mix_net.parameters())
        self.eval_hidden = None
        self.target_hidden = None
        if self.wqmix > 0:
            self.qstar_target_rnn.load_state_dict(self.qstar_eval_rnn.state_dict())
            self.qstar_target_mix.load_state_dict(self.qstar_eval_mix.state_dict())
            # self.qstar_params = list(self.qstar_eval_rnn.parameters()) + list(self.qstar_eval_mix.parameters())
            self.eval_params += list(self.qstar_eval_rnn.parameters()) + list(self.qstar_eval_mix.parameters())
            self.qstar_eval_hidden = None
            self.qstar_target_hidden = None
        if args.optim == 'RMS':
            self.optimizer = torch.optim.RMSprop(self.eval_params, lr=args.lr)
            # if self.wqmix > 0:
            #     self.qstar_optimizer = torch.optim.RMSprop(self.qstar_params, lr=args.lr)
        else:
            self.optimizer = torch.optim.Adam(self.eval_params)
            # if self.wqmix > 0:
            #     self.qstar_optimizer = torch.optim.Adam(self.qstar_params)

    def learn(self, batch, max_episode_len, train_step):
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        # for key in batch.keys():
        #     if key == 'a':
        #         batch[key] = torch.LongTensor(batch[key])
        #     else:
        #         batch[key] = torch.Tensor(batch[key])
        for key in batch.keys():
            if key == 'a':
                batch[key] = torch.as_tensor(batch[key], dtype=torch.long, device=self.args.device)
            else:
                batch[key] = torch.as_tensor(batch[key], dtype=torch.float, device=self.args.device)

        s, next_s, a, r, avail_a, next_avail_a, done = batch['s'], batch['next_s'], batch['a'], \
                                                       batch['r'], batch['avail_a'], batch['next_avail_a'], \
                                                       batch['done']
        mask = 1 - batch["padded"].float()
        eval_qs, target_qs = self.get_q(batch, episode_num, max_episode_len)
        # if self.args.cuda:
        #     a = a.cuda()
        #     r = r.cuda()
        #     done = done.cuda()
        #     mask = mask.cuda()
        #     # if 'qmix' in self.args.alg:
        #     s = s.cuda()
        #     next_s = next_s.cuda()
        eval_qsa = torch.gather(eval_qs, dim=3, index=a).squeeze(3)
        if self.args.alg == 'qatten':
            eval_q_total, q_attend_regs, head_entropies = self.eval_mix_net(eval_qsa, s, a)
        else:
            eval_q_total = self.eval_mix_net(eval_qsa, s)
        qstar_q_total, qstar_loss, q_attend_regs = None, None, None
        target_qs[next_avail_a == 0.0] = -9999999
        target_qsa = target_qs.max(dim=3)[0]
        if self.wqmix > 0:
            argmax_u = target_qs.argmax(dim=3).unsqueeze(3)
            qstar_eval_qs, qstar_target_qs = self.get_q(batch, episode_num, max_episode_len, True)
            qstar_eval_qs = torch.gather(qstar_eval_qs, dim=3, index=a).squeeze(3)
            qstar_target_qs = torch.gather(qstar_target_qs, dim=3, index=argmax_u).squeeze(3)
            qstar_q_total = self.qstar_eval_mix(qstar_eval_qs, s)
            next_q_total = self.qstar_target_mix(qstar_target_qs, next_s)
        elif self.args.alg == 'qatten':
            # chosen_action_qvals, q_attend_regs, head_entropies = self.mixer(chosen_action_qvals, batch["state"][:, :-1],
            #                                                                 actions)
            target_next_actions = target_qs.max(dim=3)[1].unsqueeze(-1).detach()
            next_q_total, q_attend_regs, _ = self.target_mix_net(target_qsa, next_s, target_next_actions)
        else:
            # target_qs[next_avail_a == 0.0] = float('-inf')
            # target_qs = target_qs.max(dim=3)[0]
            next_q_total = self.target_mix_net(target_qsa, next_s)

        target_q_total = r + self.args.gamma * next_q_total * (1 - done)
        # weights = torch.Tensor(np.ones(eval_q_total.shape))
        weights = torch.as_tensor(np.ones(eval_q_total.shape), dtype=torch.float, device=self.args.device)
        if self.wqmix > 0:
            # weights = torch.Tensor(1 - np.random.ranf(eval_q_total.shape))
            weights = torch.full(eval_q_total.shape, self.alpha, device=self.args.device)
            if self.args.alg == 'cwqmix':
                error = mask * (target_q_total - qstar_q_total)
            elif self.args.alg == 'owqmix':
                error = mask * (target_q_total - eval_q_total)
            else:
                raise Exception("")
            weights[error > 0] = 1.
            qstar_error = mask * (qstar_q_total - target_q_total.detach())

            qstar_loss = (qstar_error ** 2).sum() / mask.sum()
            # self.qstar_optimizer.zero_grad()
            # qstar_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.qstar_params, self.args.clip_norm)
            # self.qstar_optimizer.step()

        td_error = mask * (eval_q_total - target_q_total.detach())
        # if self.args.cuda:
        #     weights = weights.cuda()

        loss = (weights.detach() * td_error**2).sum() / mask.sum()
        if self.args.alg == 'qatten':
            loss += q_attend_regs
        elif self.wqmix > 0:
            loss += qstar_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_params, self.args.clip_norm)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_period == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
            if self.wqmix > 0:
                self.qstar_target_rnn.load_state_dict(self.qstar_eval_rnn.state_dict())
                self.qstar_target_mix.load_state_dict(self.qstar_eval_mix.state_dict())

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.args.n_agents, self.args.rnn_hidden_dim),
                                       device=self.args.device)
        self.target_hidden = torch.zeros((episode_num, self.args.n_agents, self.args.rnn_hidden_dim),
                                         device=self.args.device)
        if self.wqmix > 0:
            self.qstar_eval_hidden = torch.zeros((episode_num, self.args.n_agents, self.args.rnn_hidden_dim),
                                                 device=self.args.device)
            self.qstar_target_hidden = torch.zeros((episode_num, self.args.n_agents, self.args.rnn_hidden_dim),
                                                   device=self.args.device)

    def get_q(self, batch, episode_num, max_episode_len, wqmix=False):
        eval_qs, target_qs = [], []
        for trans_idx in range(max_episode_len):
            inputs, next_inputs = self.get_inputs(batch, episode_num, trans_idx)
            # if self.args.cuda:
            #     # inputs = inputs.cuda()
            #     # next_inputs = next_inputs.cuda()
            #     if wqmix:
            #         self.qstar_eval_hidden = self.qstar_eval_hidden.cuda()
            #         self.qstar_target_hidden = self.qstar_target_hidden.cuda()
            #     else:
            #         self.eval_hidden = self.eval_hidden.cuda()
            #         self.target_hidden = self.target_hidden.cuda()
            if wqmix:
                eval_q, self.qstar_eval_hidden = self.qstar_eval_rnn(inputs, self.qstar_eval_hidden)
                target_q, self.qstar_target_hidden = self.qstar_target_rnn(next_inputs, self.qstar_target_hidden)
            else:
                eval_q, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
                target_q, self.target_hidden = self.target_rnn(next_inputs, self.target_hidden)
            eval_q = eval_q.view(episode_num, self.args.n_agents, -1)
            target_q = target_q.view(episode_num, self.args.n_agents, -1)
            eval_qs.append(eval_q)
            target_qs.append(target_q)
        eval_qs = torch.stack(eval_qs, dim=1)
        target_qs = torch.stack(target_qs, dim=1)
        return eval_qs, target_qs

    def get_inputs(self, batch, episode_num, trans_idx):
        obs, next_obs, onehot_a = batch['o'][:, trans_idx], \
                                  batch['next_o'][:, trans_idx], batch['onehot_a'][:]
        inputs, next_inputs = [], []
        inputs.append(obs)
        next_inputs.append(next_obs)
        if self.args.last_action:
            if trans_idx == 0:
                inputs.append(torch.zeros_like(onehot_a[:, trans_idx], device=self.args.device))
            else:
                inputs.append(onehot_a[:, trans_idx-1])
            next_inputs.append(onehot_a[:, trans_idx])
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents, device=self.args.device).unsqueeze(0).expand(episode_num, -1, -1))
            next_inputs.append(torch.eye(self.args.n_agents, device=self.args.device).unsqueeze(0).expand(episode_num, -1, -1))
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        next_inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in next_inputs], dim=1)
        return inputs, next_inputs

    def save_model(self, train_step):
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        if type(train_step) == str:
            num = train_step
        else:
            num = str(train_step // self.args.save_model_period)

        torch.save(self.eval_mix_net.state_dict(), self.args.model_dir + '/' + num + '_'
                   + self.args.alg + '_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(), self.args.model_dir + '/' + num + '_rnn_params.pkl')
