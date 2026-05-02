

import sys

import torch
import os

from torch import nn

from network.base_net import RNN
from network.qmix_mixer import advQMIXMixer
from network.vdn_mixer import VDNMixer
from network.wqmix_q_star import QStar
from network.qatten_mixer import QattenMixer
import numpy as np


class Q_Decom:
    def __init__(self, args, itr, load_model=False):
        self.args = args
        self.load_model = load_model
        self.load_dir = ""
        input_shape = self.args.obs_shape

        if args.last_action:
            input_shape += self.args.n_actions
        if args.reuse_network:
            input_shape += self.args.n_agents

        self.args.n_actions = 6
        self.eval_rnn = RNN(input_shape, args)
        self.target_rnn = RNN(input_shape, args)

        self.wqmix = 0
        if self.args.alg == 'cwqmix' or self.args.alg == 'owqmix':
            self.wqmix = 1


        if 'qmix' in self.args.alg:
            self.eval_mix_net = advQMIXMixer(args)
            self.target_mix_net = advQMIXMixer(args)
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
                    raise Exception('Unknown algorithm')
        elif self.args.alg == 'vdn':
            self.eval_mix_net = VDNMixer()
            self.target_mix_net = VDNMixer()
        elif self.args.alg == 'qatten':
            self.eval_mix_net = QattenMixer(args)
            self.target_mix_net = QattenMixer(args)


        if args.cuda:

            self.eval_rnn, self.target_rnn, self.eval_mix_net, self.target_mix_net =\
                self.eval_rnn.to(args.device), self.target_rnn.to(args.device), self.eval_mix_net.to(
                    args.device), self.target_mix_net.to(args.device)
            self.eval_rnn, self.target_rnn, self.eval_mix_net, self.target_mix_net =\
                nn.parallel.DistributedDataParallel(self.eval_rnn, device_ids=[args.local_rank]),\
                nn.parallel.DistributedDataParallel(self.target_rnn, device_ids=[args.local_rank]),\
                nn.parallel.DistributedDataParallel(self.eval_mix_net, device_ids=[args.local_rank]),\
                nn.parallel.DistributedDataParallel(self.target_mix_net, device_ids=[args.local_rank])





            if self.wqmix > 0:
                self.qstar_eval_mix, self.qstar_target_mix, self.qstar_eval_rnn, self.qstar_target_rnn =\
                    self.qstar_eval_mix.to(args.device), self.qstar_target_mix.to(args.device), self.qstar_eval_rnn.to(
                        args.device), self.qstar_target_rnn.to(args.device)
                self.qstar_eval_mix, self.qstar_target_mix, self.qstar_eval_rnn, self.qstar_target_rnn =\
                    nn.parallel.DistributedDataParallel(self.qstar_eval_mix, device_ids=[args.local_rank]),\
                    nn.parallel.DistributedDataParallel(self.qstar_target_mix, device_ids=[args.local_rank]),\
                    nn.parallel.DistributedDataParallel(self.qstar_eval_rnn, device_ids=[args.local_rank]),\
                    nn.parallel.DistributedDataParallel(self.qstar_target_rnn, device_ids=[args.local_rank])








        if self.load_model:
            if os.path.exists(self.load_dir + '/rnn_net_params.pkl'):
                path_rnn = ""
                path_mix = ""

                map_location = 'cuda:0' if args.cuda else 'cpu'

                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_mix_net.load_state_dict(torch.load(path_mix, map_location=map_location))
                if self.wqmix > 0:
                    path_agent_rnn = ""
                    path_qstar = ""
                    self.qstar_eval_rnn.load_state_dict(torch.load(path_agent_rnn, map_location=map_location))
                    self.qstar_eval_mix.load_state_dict(torch.load(path_qstar, map_location=map_location))
                print('Successfully loaded models %s' % path_rnn + ' and %s' % path_mix)
            else:
                raise Exception("Model does not exist")

        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())

        self.eval_params = list(self.eval_rnn.parameters()) + list(self.eval_mix_net.parameters())

        self.eval_hidden = None
        self.target_hidden = None
        if self.wqmix > 0:

            self.qstar_target_rnn.load_state_dict(self.qstar_eval_rnn.state_dict())
            self.qstar_target_mix.load_state_dict(self.qstar_eval_mix.state_dict())


            self.eval_params += list(self.qstar_eval_rnn.parameters()) + list(self.qstar_eval_mix.parameters())

            self.qstar_eval_hidden = None
            self.qstar_target_hidden = None

        if args.optim == 'RMS':
            self.optimizer = torch.optim.RMSprop(self.eval_params, lr=args.lr)


        else:
            self.optimizer = torch.optim.Adam(self.eval_params)


        print("Value decomposition algorithm " + self.args.alg + " initialized")

    def learn(self, batch, max_episode_len, train_step):


        episode_num = batch['o'].shape[0]

        self.init_hidden(episode_num)






        for key in batch.keys():
            if key == 'a':
                batch[key] = torch.as_tensor(batch[key], dtype=torch.long, device=self.args.device)
            else:
                batch[key] = torch.as_tensor(batch[key], dtype=torch.float, device=self.args.device)

        s, next_s, a, r, avail_a, next_avail_a, done = batch['s'], batch['next_s'], batch['a'],\
                                                       batch['r'], batch['avail_a'], batch['next_avail_a'],\
                                                       batch['done']

        mask = 1 - batch["padded"].float()
        avail_a = avail_a[:, :, :, :6]
        next_avail_a = next_avail_a[:, :, :, :6]

        eval_qs, target_qs = self.get_q(batch, episode_num, max_episode_len)










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


            target_next_actions = target_qs.max(dim=3)[1].unsqueeze(-1).detach()
            next_q_total, q_attend_regs, _ = self.target_mix_net(target_qsa, next_s, target_next_actions)
        else:




            next_q_total = self.target_mix_net(target_qsa, next_s)

        target_q_total = r + self.args.gamma * next_q_total * (1 - done)


        weights = torch.as_tensor(np.ones(eval_q_total.shape), dtype=torch.float, device=self.args.device)
        if self.wqmix > 0:



            weights = torch.full(eval_q_total.shape, self.alpha, device=self.args.device)
            if self.args.alg == 'cwqmix':
                error = mask * (target_q_total - qstar_q_total)
            elif self.args.alg == 'owqmix':
                error = mask * (target_q_total - eval_q_total)
            else:
                raise Exception("Model does not exist")
            weights[error > 0] = 1.

            qstar_error = mask * (qstar_q_total - target_q_total.detach())

            qstar_loss = (qstar_error ** 2).sum() / mask.sum()







        td_error = mask * (eval_q_total - target_q_total.detach())



        loss = (weights.detach() * td_error**2).sum() / mask.sum()
        if self.args.alg == 'qatten':
            loss += q_attend_regs
        elif self.wqmix > 0:
            loss += qstar_loss


        sys.stdout.flush()
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

        self.eval_hidden = torch.zeros((episode_num, self.args.n_adv, self.args.rnn_hidden_dim),
                                       device=self.args.device)
        self.target_hidden = torch.zeros((episode_num, self.args.n_adv, self.args.rnn_hidden_dim),
                                         device=self.args.device)
        if self.wqmix > 0:
            self.qstar_eval_hidden = torch.zeros((episode_num, self.args.n_adv, self.args.rnn_hidden_dim),
                                                 device=self.args.device)
            self.qstar_target_hidden = torch.zeros((episode_num, self.args.n_adv, self.args.rnn_hidden_dim),
                                                   device=self.args.device)

    def get_q(self, batch, episode_num, max_episode_len, wqmix=False):
        eval_qs, target_qs = [], []
        for trans_idx in range(max_episode_len):

            inputs, next_inputs = self.get_inputs(batch, episode_num, trans_idx)











            if wqmix:
                eval_q, self.qstar_eval_hidden = self.qstar_eval_rnn(inputs, self.qstar_eval_hidden)
                target_q, self.qstar_target_hidden = self.qstar_target_rnn(next_inputs, self.qstar_target_hidden)
            else:
                eval_q, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
                target_q, self.target_hidden = self.target_rnn(next_inputs, self.target_hidden)

            eval_q = eval_q.view(episode_num, self.args.n_adv, -1)
            target_q = target_q.view(episode_num, self.args.n_adv, -1)

            eval_qs.append(eval_q)
            target_qs.append(target_q)

        eval_qs = torch.stack(eval_qs, dim=1)
        target_qs = torch.stack(target_qs, dim=1)
        return eval_qs, target_qs

    def get_inputs(self, batch, episode_num, trans_idx):

        obs, next_obs, onehot_a = batch['o'][:, trans_idx],\
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
            pass
            agents_pos = torch.zeros([self.args.n_adv, self.args.n_agents], device=self.args.device)
            for i in range(self.args.n_adv):
                agents_pos[i, -i-1] = 1
            inputs.append(agents_pos.unsqueeze(0).expand(episode_num, -1, -1))
            next_inputs.append(agents_pos.unsqueeze(0).expand(episode_num, -1, -1))

        inputs = torch.cat([x.reshape(episode_num * self.args.n_adv, -1) for x in inputs], dim=1)
        next_inputs = torch.cat([x.reshape(episode_num * self.args.n_adv, -1) for x in next_inputs], dim=1)
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
