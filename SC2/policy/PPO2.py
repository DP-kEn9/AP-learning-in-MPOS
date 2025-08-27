import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import os


METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
    dict(name='clip', epsilon=0.2),
][1]


class ActorNet(nn.Module):
    def __init__(self, args, n_hiddens=100):
        super(ActorNet, self).__init__()
        input_shape = args.obs_shape
        if args.last_action:
            input_shape += args.n_actions
        if args.reuse_network:
            input_shape += args.n_agents
        self.fc1 = nn.Linear(input_shape, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, 6)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


class CriticNet(nn.Module):
    def __init__(self, args, output=1):
        super(CriticNet, self).__init__()
        input_shape = args.state_shape
        # if args.last_action:
        #     input_shape += args.n_actions
        # if args.reuse_network:
        #     input_shape += args.n_agents
        self.in_to_y1 = nn.Linear(input_shape, 100)
        self.in_to_y1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(100, output)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, inputstate):
        inputstate = self.in_to_y1(inputstate)
        inputstate = F.relu(inputstate)
        Q = self.out(inputstate)
        return Q


class Actor():
    def __init__(self, args):
        self.args = args
        self.n_action = args.n_actions
        self.old_pi, self.new_pi = ActorNet(args), ActorNet(args)  # 这只是均值mean
        self.optimizer = torch.optim.Adam(self.new_pi.parameters(), lr=args.A_LR, eps=1e-5)
        if args.cuda:
            # torch.distributed.init_process_group(backend="nccl")
            self.old_pi, self.new_pi = self.old_pi.to(args.device), self.new_pi.to(args.device)
            self.old_pi, self.new_pi = \
                nn.parallel.DistributedDataParallel(self.old_pi, device_ids=[args.local_rank]), \
                    nn.parallel.DistributedDataParallel(self.new_pi, device_ids=[args.local_rank])

    def choose_action(self, inputstate, avail_actions):
        # inputstate = torch.FloatTensor(s)
        probs = self.old_pi(inputstate)
        probs = probs[0] + 1e-6
        avail_a = np.zeros(6)
        avail_a[avail_actions] = 1
        probs[np.where(avail_a == 0)] = 0
        probs = probs / torch.sum(probs)
        # mean. std = mean.cpu(), std.cpu()
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        # action = torch.clamp(action, 0, avail_actions[-1])
        # avail_actions = np.array(avail_actions)
        # action = avail_actions[np.argmin(np.abs(avail_actions - int(action[0][0])))]
        # action_T = torch.Tensor([[action]])
        # action_T = action_T.cuda(device=self.args.local_rank)
        if self.args.device != 'cpu':
            action = action.cuda(device=self.args.local_rank)
        action_logprob = dist.log_prob(action)
        # if action_logprob.item() is torch.nan:
        #     print(1111)
        return action, action_logprob.item()

    def update_oldpi(self):
        self.old_pi.load_state_dict(self.new_pi.state_dict())

    def get_loss(self, bs, ba, adv, bap, buffer_av):
        ba = ba.unsqueeze(1)
        bav = torch.FloatTensor(buffer_av)
        if self.args.device != 'cpu':
            bs = bs.cuda(device=self.args.local_rank)
            bav = bav.cuda(device=self.args.local_rank)
            ba = ba.cuda(device=self.args.local_rank)
            bap = bap.cuda(device=self.args.local_rank)
            adv = adv.cuda(device=self.args.local_rank)
        probs = self.new_pi(bs)
        mask_probs = torch.add(torch.mul(probs, bav), 1e-6)
        mask_probs_normal = mask_probs / torch.sum(mask_probs)
        dist_new = torch.distributions.Categorical(mask_probs_normal)
        action_new_logprob = dist_new.log_prob(ba[:, 0])
        ratio = torch.exp(action_new_logprob - bap.detach())
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - METHOD['epsilon'], 1 + METHOD['epsilon']) * adv
        loss = -torch.min(surr1, surr2)
        return loss


    # def learn(self, bs, ba, adv, bap, buffer_av):
    #     # bs = torch.FloatTensor(bs)
    #     # ba = torch.FloatTensor(ba)
    #     # adv = torch.FloatTensor(adv)
    #     # bap = torch.FloatTensor(bap)
    #     ba = ba.unsqueeze(1)
    #     bav = torch.FloatTensor(buffer_av)
    #     if self.args.device != 'cpu':
    #         bs = bs.cuda(device=self.args.local_rank)
    #         bav = bav.cuda(device=self.args.local_rank)
    #         ba = ba.cuda(device=self.args.local_rank)
    #         bap = bap.cuda(device=self.args.local_rank)
    #         adv = adv.cuda(device=self.args.local_rank)
    #     for _ in range(self.args.A_UPDATE_STEPS):
    #         probs = self.new_pi(bs)
    #         mask_probs = torch.add(torch.mul(probs, bav), 1e-6)
    #         mask_probs_normal = mask_probs / torch.sum(mask_probs)
    #         dist_new = torch.distributions.Categorical(mask_probs_normal)
    #         action_new_logprob = dist_new.log_prob(ba[:, 0])
    #         ratio = torch.exp(action_new_logprob - bap.detach())
    #         surr1 = ratio * adv
    #         surr2 = torch.clamp(ratio, 1 - METHOD['epsilon'], 1 + METHOD['epsilon']) * adv
    #         loss = -torch.min(surr1, surr2)
    #         loss = loss.mean()
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         nn.utils.clip_grad_norm_(self.new_pi.parameters(), 0.5)
    #         self.optimizer.step()

    def learn(self, loss):
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.new_pi.parameters(), 0.5)
        self.optimizer.step()


class Critic():
    def __init__(self, args):
        self.args = args
        self.critic_v = CriticNet(args, 1)
        self.optimizer = torch.optim.Adam(self.critic_v.parameters(), lr=args.C_LR, eps=1e-5)
        self.lossfunc = nn.MSELoss()
        if self.args.device != 'cpu':
            # torch.distributed.init_process_group(backend="nccl")
            self.critic_v = self.critic_v.to(args.device)
            self.critic_v = nn.parallel.DistributedDataParallel(self.critic_v, device_ids=[args.local_rank])

    def get_v(self, s):
        inputstate = torch.FloatTensor(s)
        return self.critic_v(inputstate)

    # def get_adv(self,bs,br):
    #     reality_v=torch.FloatTensor(br)
    #     v=self.get_v(bs)
    #     adv=(reality_v-v).detach()
    #     return adv

    # def learn(self, bs, reality_v):
    #     # bs = torch.FloatTensor(bs)
    #     # reality_v = torch.FloatTensor(br)
    #     reality_v = reality_v.unsqueeze(1)
    #     # if self.args.device != 'cpu':
    #     #     reality_v.cuda(device=self.args.local_rank)
    #     #     bs.cuda(device=self.args.local_rank)
    #     for _ in range(self.args.C_UPDATE_STEPS):
    #         v = self.get_v(bs)
    #         if self.args.device != 'cpu':
    #             v = v.cuda(device=self.args.local_rank)
    #             reality_v = reality_v.cuda(device=self.args.local_rank)
    #         # print(reality_v.device, v.device)
    #         td_e = self.lossfunc(reality_v, v)
    #         self.optimizer.zero_grad()
    #         td_e.backward()
    #         nn.utils.clip_grad_norm_(self.critic_v.parameters(), 0.5)
    #         self.optimizer.step()
    #     return (reality_v - v).detach()
    def learn(self, reality_v, v):
        td_e = self.lossfunc(reality_v, v)
        self.optimizer.zero_grad()
        td_e.backward()
        nn.utils.clip_grad_norm_(self.critic_v.parameters(), 0.5)
        self.optimizer.step()


class PPO():
    def __init__(self, args):
        self.actor = Actor(args)
        self.critic = Critic(args)
        self.args = args

    def choose_action(self, obs, avail_actions):
        action, action_logprob = self.actor.choose_action(obs, avail_actions)
        return action, action_logprob

    def learn(self, batch, max_episode_len, train_step):
        batch_v, batch_rv, batch_loss = None, None, None
        for i in range(self.args.batch_size):
            buffer_s, buffer_a, buffer_r, buffer_ap, buffer_av, buffer_o = batch['s'][i, :], batch['a'][i, :, -1, -1], \
                batch['r'][i, :], batch['ap'][i, :, 0], batch['avail_a'][i, :, -1, :6], \
                batch['o'][i, :, -1, :]
            buffer_la = np.zeros((buffer_a.shape[0], self.args.n_actions))
            for j in range(buffer_a.shape[0]):
                buffer_la[j, int(buffer_a[j])] = 1
            la_0 = np.zeros((1, self.args.n_actions))
            buffer_la = np.vstack((buffer_la, la_0))[:-1, :]
            buffer_idx = np.hstack((np.zeros((buffer_a.shape[0], self.args.n_agents - 1)),
                                    np.ones((buffer_a.shape[0], 1))))
            buffer_o = np.hstack((buffer_o, buffer_la, buffer_idx))
            o_ = buffer_s[-1, :]
            v_ = self.critic.get_v(o_)
            discounted_r = []
            for reward in buffer_r[::-1]:
                reward = torch.Tensor(reward)
                if self.args.device != 'cpu':
                    reward = reward.cuda(device=self.args.local_rank)
                v_ = reward + 0.9 * v_
                discounted_r.append(v_)
            bs, ba, br, bap, bo = torch.FloatTensor(buffer_s), torch.FloatTensor(buffer_a), \
                torch.FloatTensor(discounted_r), torch.FloatTensor(buffer_ap), torch.FloatTensor(buffer_o)
            reality_v = br.unsqueeze(1)
            v = self.critic.get_v(bs)
            if self.args.device != 'cpu':
                v = v.cuda(device=self.args.local_rank)
                reality_v = reality_v.cuda(device=self.args.local_rank)
            advantage = (reality_v - v).detach()
            actor_loss = self.actor.get_loss(bo, ba, advantage, bap, buffer_av)
            if batch_v is None:
                batch_v = v
                batch_rv = reality_v
                batch_loss = actor_loss
            else:
                batch_v = torch.vstack([batch_v, v])
                batch_rv = torch.vstack([batch_rv, reality_v])
                batch_loss = torch.vstack([batch_loss, actor_loss])
        self.critic.learn(batch_rv, batch_v)
        self.actor.learn(batch_loss)
        self.actor.update_oldpi()
            # advantage = self.critic.learn(bs, br)
            # self.actor.learn(bo, ba, advantage, bap, buffer_av)
            # self.actor.update_oldpi()
            # self.critic.learn(bs, br)

    def save_model(self, train_step):
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        if type(train_step) == str:
            num = train_step
        else:
            num = str(train_step // self.args.save_model_period)

        torch.save(self.actor.old_pi.state_dict(),
                   self.args.model_dir + '/' + num + '_' + self.args.adv_alg + '_actor.pkl')
        torch.save(self.critic.critic_v.state_dict(), self.args.model_dir + '/' + num + '_critic.pkl')

    def init_hidden(self, i):
        return 123
