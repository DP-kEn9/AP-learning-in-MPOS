import torch
import os
from network.base_net import RNN
from network.dmaq_general import DMAQer
from network.dmaq_qatten import DMAQ_QattenMixer
import torch.nn as nn
import numpy as np


class DMAQ_qattenLearner:
    def __init__(self, args, itr):
        self.args = args
        input_shape = self.args.obs_shape

        if args.last_action:
            input_shape += self.args.n_actions
        if args.reuse_network:
            input_shape += self.args.n_agents

        self.eval_rnn = RNN(input_shape, args)
        self.target_rnn = RNN(input_shape, args)


        if self.args.alg == 'dmaq_qatten':
            self.mixer = DMAQ_QattenMixer()
            self.target_mixer = DMAQ_QattenMixer()
        elif self.args.alg == 'dmaq':
            self.mixer = DMAQer(args)
            self.target_mixer = DMAQer(args)
        else:
            raise Exception('Unsupported!')


        if args.cuda:
            self.eval_rnn, self.target_rnn, self.mixer, self.target_mixer =\
                nn.DataParallel(self.eval_rnn), nn.DataParallel(self.target_rnn), nn.DataParallel(self.mixer),\
                nn.DataParallel(self.target_mixer)
            self.eval_rnn, self.target_rnn, self.mixer, self.target_mixer =\
                self.eval_rnn.to(args.device), self.target_rnn.to(args.device), self.mixer.to(args.device), self.target_mixer.to(args.device)







        if args.load_model:
            if os.path.exists(self.args.model_dir + '/rnn_net_params.pkl'):
                path_rnn = ""
                path_mix = ""

                map_location = 'cuda:0' if args.cuda else 'cpu'

                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.mixer.load_state_dict(torch.load(path_mix, map_location=map_location))

                print('Successfully loaded models %s' % path_rnn + ' and %s' % path_mix)
            else:
                raise Exception("Model does not exist")

        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.eval_params = list(self.eval_rnn.parameters()) + list(self.mixer.parameters())

        self.eval_hidden = None
        self.target_hidden = None


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
                batch[key] = torch.LongTensor(batch[key])
            else:
                batch[key] = torch.Tensor(batch[key])
        s, next_s, a, r, avail_a, next_avail_a, done, actions_onehot = batch['s'], batch['next_s'], batch['a'],\
                                                                       batch['r'], batch['avail_a'], batch[
                                                                           'next_avail_a'],\
                                                                       batch['done'], batch['onehot_a']

        mask = 1 - batch["padded"].float()

        eval_qs, target_qs = self.get_q(batch, episode_num, max_episode_len)

        if self.args.cuda:
            a = a.cuda()
            r = r.cuda()
            done = done.cuda()
            mask = mask.cuda()
            s = s.cuda()
            next_s = next_s.cuda()
            actions_onehot = actions_onehot.cuda()

        eval_qsa = torch.gather(eval_qs, dim=3, index=a).squeeze(3)
        max_action_qvals = eval_qs.max(dim=3)[0]

        target_qs[next_avail_a == 0.0] = -9999999
        target_qsa = target_qs.max(dim=3)[0]


        q_attend_regs = None
        if self.args.alg == "dmaq_qatten":
            ans_chosen, q_attend_regs, head_entropies = self.mixer(eval_qsa, s, is_v=True)
            ans_adv, _, _ = self.mixer(eval_qsa, s, actions=actions_onehot,
                                       max_q_i=max_action_qvals, is_v=False)
            eval_qsa = ans_chosen + ans_adv
        else:
            ans_chosen = self.mixer(eval_qsa, s, is_v=True)
            ans_adv = self.mixer(eval_qsa, s, actions=actions_onehot,
                                 max_q_i=max_action_qvals, is_v=False)
            eval_qsa = ans_chosen + ans_adv















        target_max_qvals = self.target_mixer(target_qsa, next_s, is_v=True)


        targets = r + self.args.gamma * (1 - done) * target_max_qvals


        td_error = (eval_qsa - targets.detach())

        mask = mask.expand_as(td_error)


        masked_td_error = td_error * mask


        loss = (masked_td_error ** 2).sum() / mask.sum()
        if self.args.alg == "dmaq_qatten":
            loss += q_attend_regs



        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_params, self.args.clip_norm)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_period == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def init_hidden(self, episode_num):

        self.eval_hidden = torch.zeros((episode_num, self.args.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.args.n_agents, self.args.rnn_hidden_dim))

    def get_q(self, batch, episode_num, max_episode_len, ):
        eval_qs, target_qs = [], []
        for trans_idx in range(max_episode_len):

            inputs, next_inputs = self.get_inputs(batch, episode_num, trans_idx)

            if self.args.cuda:
                inputs = inputs.cuda()
                next_inputs = next_inputs.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()

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

        obs, next_obs, onehot_a = batch['o'][:, trans_idx],\
                                  batch['next_o'][:, trans_idx], batch['onehot_a'][:]
        inputs, next_inputs = [], []
        inputs.append(obs)
        next_inputs.append(next_obs)

        if self.args.last_action:
            if trans_idx == 0:
                inputs.append(torch.zeros_like(onehot_a[:, trans_idx]))
            else:
                inputs.append(onehot_a[:, trans_idx - 1])
            next_inputs.append(onehot_a[:, trans_idx])
        if self.args.reuse_network:
            pass
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            next_inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        next_inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in next_inputs], dim=1)
        return inputs, next_inputs

    def save_model(self, train_step):
        print(self.args.model_dir)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        if type(train_step) == str:
            num = train_step
        else:
            num = str(train_step // self.args.save_model_period)

        torch.save(self.mixer.state_dict(), self.args.model_dir + '/' + num + '_'
                   + self.args.alg + '_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(), self.args.model_dir + '/' + num + '_rnn_params.pkl')




































































































































































































