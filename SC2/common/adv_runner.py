import os
import random

import numpy as np
from common.adv_buffer import AdvBuffer as ReplayBuffer
from adv_agents import AdvAgents
from agents import Agents
import time
import torch
from datetime import datetime, timedelta, timezone
from network.base_net import Reward_net
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, env, args, itr, seed):
        if seed is not None:
            self.setup_seed(seed)
        self.args = args

        self.env = env
        self.pid = itr

        self.replay_buffer = ReplayBuffer(self.args)

        self.win_rates = []
        self.store = True
        self.ally_death_num = 0
        self.episodes_rewards = []
        self.evaluate_itr = []

        self.min_win_rate = 0
        self.time_steps = 0

        alg_dir = self.args.alg + '_' + str(self.args.epsilon_anneal_steps // 10000) + 'w' + '_' + \
                  str(self.args.target_update_period)
        self.alg_tag = '_' + self.args.optim

        if self.args.her:
            self.alg_tag += str(self.args.her)
            alg_dir += '_her=' + str(self.args.her)

        # self.save_path = self.args.result_dir + '/' + alg_dir + '/' + self.args.map + '/' + itr
        self.save_path = self.args.result_dir + '/' + self.args.map + '/' + alg_dir + '/' + itr
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.args.model_dir = args.model_dir + '/' + args.map + '/' + alg_dir + '/' + itr

        self.agents = Agents(args, itr=itr)
        # args.load_model = False
        self.adv_agent = AdvAgents(args, itr=itr)
        self.reward_net = Reward_net(args.obs_shape, args.rnn_hidden_dim)
        self.hidden = torch.zeros(args.rnn_hidden_dim)
        self.optimizer = torch.optim.Adam(self.reward_net.parameters())
        self.loss_func = torch.nn.MSELoss(reduction="sum")
        print('step runner init')
        if self.args.her:
            print('HER')

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def generate_episode(self, episode_num, evaluate=False, pre_train=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:
            self.env.close()
        self.env.reset()
        done = False
        info = None
        win = False
        random_action = False
        if random.random() < 0.5:
            random_action = True

        last_action = np.zeros((self.args.n_agents + 1, self.args.n_actions))
        epsilon = 0 if evaluate else self.args.epsilon
        if self.args.epsilon_anneal_scale == 'episode' or \
                (self.args.epsilon_anneal_scale == 'itr' and episode_num == 0):
            epsilon = epsilon - self.args.epsilon_decay if epsilon > self.args.min_epsilon else epsilon

        episode_buffer = None
        reward_all = None
        if not evaluate:
            episode_buffer = {'o': np.zeros([self.args.episode_limit, self.args.n_adv, self.args.obs_shape]),
                              's': np.zeros([self.args.episode_limit, self.args.state_shape]),
                              'a': np.zeros([self.args.episode_limit, self.args.n_adv, 1]),
                              'onehot_a': np.zeros([self.args.episode_limit, self.args.n_adv, self.args.n_actions]),
                              'avail_a': np.zeros([self.args.episode_limit, self.args.n_adv, self.args.n_actions]),
                              'r': np.zeros([self.args.episode_limit, 1]),
                              'r_m': np.zeros([self.args.episode_limit, 1]),
                              'next_o': np.zeros([self.args.episode_limit, self.args.n_adv, self.args.obs_shape]),
                              'next_s': np.zeros([self.args.episode_limit, self.args.state_shape]),
                              'next_avail_a': np.zeros(
                                  [self.args.episode_limit, self.args.n_adv, self.args.n_actions]),
                              'done': np.ones([self.args.episode_limit, 1]),
                              'padded': np.ones([self.args.episode_limit, 1]),
                              'ap': np.zeros([self.args.episode_limit, 1])
                              }
        states, former_states = [], []
        obs = self.env.get_obs()
        if self.args.her:
            obs = np.concatenate((obs, self.env.goal), axis=1)
        state = self.env.get_state()
        if self.args.her:
            states.append(self.env.state)
            former_states.append(self.env.former_states)
        avail_actions = []
        self.agents.policy.init_hidden(1)
        self.adv_agent.policy.init_hidden(1)
        for agent_id in range(self.args.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)

        episode_reward = 0
        tar_qsa = 0
        for step in range(self.args.episode_limit):
            if done:
                break
            else:
                actions, onehot_actions = [], []
                for agent_id in range(self.args.n_agents):
                    if agent_id >= self.args.n_agents - self.args.n_adv:
                        if pre_train and random_action:
                            action = np.random.choice(np.nonzero(avail_actions[agent_id])[0])
                        else:
                            action, ap = self.adv_agent.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                                      avail_actions[agent_id], epsilon, evaluate)
                            if 'PPO' in self.args.adv_alg:
                                if not evaluate:
                                    # print(ap)
                                    episode_buffer['ap'][step] = ap
                        # action = np.random.choice(np.nonzero(avail_actions[agent_id])[0])
                    # avail_action = self.env.get_avail_agent_actions(agent_id)
                    else:
                        if pre_train and random_action:
                            action = np.random.choice(np.nonzero(avail_actions[agent_id])[0])
                        else:
                            action, qsa = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                                    avail_actions[agent_id], epsilon, True)
                            if agent_id == 0:
                                tar_qsa = torch.max(qsa)
                                tar_qsa = float(tar_qsa.item())
                    onehot_action = np.zeros(self.args.n_actions)
                    onehot_action[action] = 1
                    onehot_actions.append(onehot_action)
                    if type(action) == torch.Tensor:
                        action = action.cpu()
                    actions.append(action)
                    # avail_actions.append(avail_action)
                    last_action[agent_id] = onehot_action
                reward, done, info = self.env.step(actions)
                if self.args.reward_model == 0:
                    reward = self.get_adv_reward()
                else:
                    reward_m, self.hidden = self.reward_net(torch.FloatTensor(self.env.get_obs()[-1]), self.hidden)
                    reward = float(reward_m[0])
                if not evaluate:
                    self.time_steps += 1
                if not done:
                    if step == self.args.episode_limit - 1:
                        reward += 100
                    next_obs = self.env.get_obs()
                    if self.args.her:
                        next_obs = np.concatenate((next_obs, self.env.goal), axis=1)
                    next_state = self.env.get_state()
                    if self.args.her:
                        states.append(self.env.state)
                        former_states.append(self.env.former_states)
                else:
                    next_obs = obs
                    next_state = state
                    if info.__contains__('battle_won'):
                        if not info['battle_won']:
                            reward += 200
                    else:
                        reward += 100
                next_avail_actions = []
                for agent_id in range(self.args.n_agents):
                    avail_action = self.env.get_avail_agent_actions(agent_id)
                    next_avail_actions.append(avail_action)
                if not evaluate:
                    episode_buffer['o'][step] = obs[self.args.n_agents - self.args.n_adv:]
                    episode_buffer['s'][step] = state
                    # if type(actions[-1]) == torch.Tensor:
                    #     actions[-1] = actions[-1].cpu()
                    # print(type(actions[-1]))
                    episode_buffer['a'][step] = np.reshape(actions[self.args.n_agents - self.args.n_adv:],
                                                           [self.args.n_adv, 1])
                    episode_buffer['onehot_a'][step] = onehot_actions[self.args.n_agents - self.args.n_adv:]
                    episode_buffer['avail_a'][step] = avail_actions[self.args.n_agents - self.args.n_adv:]
                    episode_buffer['r'][step] = [reward]
                    if self.args.reward_model == 1:
                        if reward_all is None:
                            reward_all = reward_m
                        else:
                            reward_all = torch.cat((reward_all, reward_m))
                    episode_buffer['r_m'][step] = [reward]
                    episode_buffer['next_o'][step] = next_obs[self.args.n_agents - self.args.n_adv:]
                    episode_buffer['next_s'][step] = next_state
                    episode_buffer['next_avail_a'][step] = next_avail_actions[self.args.n_agents - self.args.n_adv:]
                    episode_buffer['done'][step] = [done]
                    episode_buffer['padded'][step] = [0.]

                episode_reward += reward
                obs = next_obs
                state = next_state
                avail_actions = next_avail_actions
                if self.args.epsilon_anneal_scale == 'step':
                    epsilon = epsilon - self.args.epsilon_decay if epsilon > self.args.min_epsilon else epsilon

        if not evaluate:
            self.args.epsilon = epsilon
        if info.__contains__('battle_won'):
            win = True if done and info['battle_won'] else False
        if self.args.reward_model == 1 and not evaluate:
            value_true = 0
            d_enemies = len(np.nonzero(self.env.death_tracker_enemy)[0])
            d_allies = len(np.nonzero(self.env.death_tracker_ally)[0])
            v_e, v_a = -d_enemies * 20, d_allies * 20
            value_true = value_true + v_e + v_a
            if abs(reward_all.sum() - value_true) < 10 and reward_all.sum()*value_true > 0:
                self.store = True
            else:
                self.store = False
            loss = self.loss_func(reward_all.sum(), torch.FloatTensor([value_true]))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), self.args.clip_norm)
            self.optimizer.step()
        if evaluate and episode_num == self.args.evaluate_num - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        if not evaluate and self.args.her:
            return episode_buffer, states, former_states
        self.hidden = torch.zeros(self.args.rnn_hidden_dim)
        if pre_train:
            return reward_all.sum(), value_true
        return episode_buffer, episode_reward, win

    def run(self):
        win_rate, episodes_reward = self.evaluate()
        print("win_rate: " + str(win_rate))
        train_steps = 0
        early_stop = 10
        num_eval = 0
        self.min_win_rate = 1
        self.time_steps = 0
        last_test_step = 0
        begin_time = None
        begin_step = None

        # for itr in range(self.args.n_itr):
        while self.time_steps < self.args.max_steps:
            if begin_step is None:
                begin_time = datetime.utcnow().astimezone(timezone(timedelta(hours=8)))
                begin_step = self.time_steps
            if self.args.her:
                episode_batch, states, former_states = self.generate_episode(0)
                self.her_k(episode_batch, states, former_states)
            else:
                episode_batch, _, _ = self.generate_episode(0)
            for key in episode_batch.keys():
                episode_batch[key] = np.array([episode_batch[key]])
            for e in range(1, self.args.n_episodes):
                if self.args.her:
                    episode_batch, states, former_states = self.generate_episode(e)
                    self.her_k(episode_batch, states, former_states)
                else:
                    episode, _, _ = self.generate_episode(e)

                for key in episode_batch.keys():
                    episode[key] = np.array([episode[key]])
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            if self.store:
                self.replay_buffer.store(episode_batch)
            if self.replay_buffer.size < self.args.batch_size * self.args.bs_rate:
                begin_time = None
                begin_step = None
                continue
            for _ in range(self.args.train_steps):
                batch = self.replay_buffer.sample(self.args.batch_size)
                self.adv_agent.train(batch, train_steps)
                train_steps += 1
            # if itr % self.args.evaluation_period == 0:
            if (self.time_steps - last_test_step) / self.args.evaluation_steps_period >= 1.0:
                num_eval += 1
                last_test_step = self.time_steps
                print(f'process {self.pid}: {self.time_steps} step / {self.args.max_steps} steps')
                win_rate, episodes_reward = self.evaluate()
                print("win_rate: " + str(win_rate))
                self.evaluate_itr.append(self.time_steps)
                self.win_rates.append(win_rate)
                self.episodes_rewards.append(episodes_reward)
                if win_rate < self.min_win_rate:
                    self.min_win_rate = win_rate
                    self.adv_agent.policy.save_model(str(win_rate))
                if num_eval % 50 == 0 or num_eval % 75 == 0:
                    self.save_results()
                    self.adv_agent.policy.save_model(str(self.time_steps))
                    self.plot()
                    now = datetime.utcnow().astimezone(timezone(timedelta(hours=8)))
                    elapsed_time = now - begin_time
                    expected_remain_time = (elapsed_time / (self.time_steps - begin_step)) * \
                                           (self.args.max_steps - self.time_steps)
                    expected_end_time = now + expected_remain_time
        self.save_results()
        self.plot()
        self.env.close()

    def evaluate(self):
        win_number = 0
        episodes_reward = []
        for itr in range(self.args.evaluate_num):
            if self.args.didactic:
                episode_reward, win = self.get_eval_qtot()
            else:
                _, episode_reward, win = self.generate_episode(itr, evaluate=True)
            episodes_reward.append(episode_reward)
            if win:
                win_number += 1
        return win_number / self.args.evaluate_num, episodes_reward

    def save_results(self):
        for filename in os.listdir(self.save_path):
            if filename.endswith('.npy'):
                os.remove(self.save_path + '/' + filename)
        np.save(self.save_path + '/evaluate_itr.npy', self.evaluate_itr)
        if self.args.didactic and self.args.power is None and 'strapped' in self.args.alg:
            np.save(self.save_path + '/train_steps.npy', self.agents.policy.train_steps)
            np.save(self.save_path + '/differences.npy', self.agents.policy.differences)
        else:
            np.save(self.save_path + '/win_rates.npy', self.win_rates)
        np.save(self.save_path + '/episodes_rewards.npy', self.episodes_rewards)

    def plot(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        if self.args.didactic and self.args.power is None and 'strapped' in self.args.alg:
            win_x = np.array(self.agents.policy.train_steps)[:, None] / 1000000.
            win_y = np.array(self.agents.policy.differences)[:, None]
            plot_win = pd.DataFrame(np.concatenate((win_x, win_y), axis=1), columns=['T (mil)', self.args.which_diff])
            sns.lineplot(x="T (mil)", y=self.args.which_diff, data=plot_win, ax=ax1)
        else:
            win_x = np.array(self.evaluate_itr)[:, None] / 1000000.
            win_y = np.array(self.win_rates)[:, None]
            plot_win = pd.DataFrame(np.concatenate((win_x, win_y), axis=1), columns=['T (mil)', 'Test Win'])
            sns.lineplot(x="T (mil)", y="Test Win", data=plot_win, ax=ax1)

        ax2 = fig.add_subplot(212)
        reward_x = np.repeat(self.evaluate_itr, self.args.evaluate_num)[:, None] / 1000000.
        reward_y = np.array(self.episodes_rewards).flatten()[:, None]
        plot_reward = pd.DataFrame(np.concatenate((reward_x, reward_y), axis=1),
                                   columns=['T (mil)', 'Median Test Returns'])
        sns.lineplot(x="T (mil)", y="Median Test Returns", data=plot_reward, ax=ax2,
                     ci='sd', estimator=np.mean)
        plt.tight_layout()
        tag = self.args.alg + '_' + str(self.args.target_update_period)
        tag += (self.alg_tag + '_' +
                datetime.utcnow().astimezone(
                    timezone(timedelta(hours=8))
                ).strftime("%Y-%m-%d_%H-%M-%S"))
        for filename in os.listdir(self.save_path):
            if filename.endswith('.png'):
                os.remove(self.save_path + '/' + filename)
        fig.savefig(self.save_path + "/%s.png" % tag)
        plt.close()

    def get_eval_qtot(self):
        self.env.reset()

        all_last_action = np.zeros((self.args.n_agents, self.args.n_actions))

        all_obs = self.env.get_obs()
        state = self.env.get_state()
        avail_actions = []
        self.agents.policy.init_hidden(1)
        eval_qs = []
        actions = []
        one_hot_actions = []
        hidden_evals = None
        for agent_idx in range(self.args.n_agents):
            obs = all_obs[agent_idx]
            last_action = all_last_action[agent_idx]
            avail_action = self.env.get_avail_agent_actions(agent_idx)
            avail_actions.append(avail_action)

            onehot_agent_idx = np.zeros(self.args.n_agents)
            onehot_agent_idx[agent_idx] = 1.
            if self.args.last_action:
                obs = np.hstack((obs, last_action))
            if self.args.reuse_network:
                obs = np.hstack((obs, onehot_agent_idx))
            hidden_state = self.agents.policy.eval_hidden[:, agent_idx, :]
            obs = torch.Tensor(obs).unsqueeze(0)
            if self.args.cuda:
                obs = obs.cuda()
                hidden_state = hidden_state.cuda()
            qsa, hidden_eval = self.agents.policy.eval_rnn(obs, hidden_state)
            qsa[avail_action == 0.0] = -float("inf")

            eval_qs.append(torch.max(qsa))

            action = torch.argmax(qsa)
            actions.append(action)

            onehot_action = np.zeros(self.args.n_actions)
            onehot_action[action] = 1
            one_hot_actions.append(onehot_action)
            if hidden_evals is None:
                hidden_evals = hidden_eval
            else:
                hidden_evals = torch.cat([hidden_evals, hidden_eval], dim=0)

        s = torch.Tensor(state)
        eval_qs = torch.Tensor(eval_qs).unsqueeze(0)
        actions = torch.Tensor(actions).unsqueeze(0)
        one_hot_actions = torch.Tensor(one_hot_actions).unsqueeze(0)
        hidden_evals = hidden_evals.unsqueeze(0)
        if self.args.cuda:
            s = s.cuda()
            eval_qs = eval_qs.cuda()
            actions = actions.cuda()
            one_hot_actions = one_hot_actions.cuda()
            hidden_evals = hidden_evals.cuda()
        eval_q_total = None
        if self.args.alg == 'qatten':
            eval_q_total, _, _ = self.agents.policy.eval_mix_net(eval_qs, s, actions)
        elif self.args.alg == 'qmix' \
                or 'wqmix' in self.args.alg \
                or 'strapped' in self.args.alg:
            eval_q_total = self.agents.policy.eval_mix_net(eval_qs, s)
        elif 'dmaq' in self.args.alg:
            if self.args.alg == "dmaq_qatten":
                ans_chosen, _, _ = self.agents.policy.mixer(eval_qs, s, is_v=True)
                ans_adv, _, _ = self.agents.policy.mixer(eval_qs, s, actions=one_hot_actions,
                                                         max_q_i=eval_qs, is_v=False)
                eval_q_total = ans_chosen + ans_adv
            else:
                ans_chosen = self.agents.policy.mixer(eval_qs, s, is_v=True)
                ans_adv = self.agents.policy.mixer(eval_qs, s, actions=one_hot_actions,
                                                   max_q_i=eval_qs, is_v=False)
                eval_q_total = ans_chosen + ans_adv
        elif self.args.alg == 'qtran_base':
            one_hot_actions = one_hot_actions.unsqueeze(0)
            hidden_evals = hidden_evals.unsqueeze(0)
            eval_q_total = self.agents.policy.eval_joint_q(s, hidden_evals, one_hot_actions)

        eval_q_total = eval_q_total.squeeze().item()
        return eval_q_total, 0

    def her_k(self, episode, states, former_states):
        import copy
        for _ in range(self.args.her):
            episode_buffer = {'o': np.zeros([self.args.episode_limit, self.args.n_agents, self.args.obs_shape]),
                              's': np.zeros([self.args.episode_limit, self.args.state_shape]),
                              'a': np.zeros([self.args.episode_limit, self.args.n_agents, 1]),
                              'onehot_a': np.zeros([self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
                              'avail_a': np.zeros([self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
                              'r': np.zeros([self.args.episode_limit, 1]),
                              'next_o': np.zeros([self.args.episode_limit, self.args.n_agents, self.args.obs_shape]),
                              'next_s': np.zeros([self.args.episode_limit, self.args.state_shape]),
                              'next_avail_a': np.zeros(
                                  [self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
                              'done': np.ones([self.args.episode_limit, 1]),
                              'padded': np.ones([self.args.episode_limit, 1])
                              }
            self.env.reset()
            for i in range(len(episode)):
                reward = self.env.get_reward(states[i], former_states[i])
                done = episode['done'][i]
                if reward >= 0:
                    reward = 0
                    done = True
                episode_buffer['o'][i] = episode['o'][i]
                episode_buffer['o'][i, :, -2:] = np.array(self.env.goal)[:]
                episode_buffer['s'][i] = episode['s'][i]
                episode_buffer['a'][i] = episode['a'][i]
                episode_buffer['onehot_a'][i] = episode['onehot_a'][i]
                episode_buffer['avail_a'][i] = episode['avail_a'][i]
                episode_buffer['r'][i] = [reward]
                episode_buffer['next_o'][i] = episode['next_o'][i]
                episode_buffer['next_o'][i, :, -2:] = np.array(self.env.goal)[:]
                episode_buffer['next_s'][i] = episode['next_s'][i]
                episode_buffer['next_avail_a'][i] = episode['next_avail_a'][i]
                episode_buffer['done'][i] = [done]
                episode_buffer['padded'][i] = [0.]
                if done:
                    break
            for key in episode_buffer.keys():
                episode_buffer[key] = np.array([episode_buffer[key]])
            self.replay_buffer.store(episode_buffer)

    def get_adv_reward(self):
        reward = 0
        delta_deaths = 0
        delta_ally = 0
        delta_enemy = 0

        neg_scale = self.env.reward_negative_scale

        # update deaths
        for al_id, al_unit in self.env.agents.items():
            if not self.env.death_tracker_ally[al_id]:
                prev_health = (
                        self.env.previous_ally_units[al_id].health
                        + self.env.previous_ally_units[al_id].shield
                )
                if al_unit.health == 0:
                    # just died
                    delta_deaths += 20 * neg_scale
                    delta_ally += prev_health * neg_scale
                else:
                    # still alive
                    delta_ally += neg_scale * (
                            prev_health - al_unit.health - al_unit.shield
                    )

        for e_id, e_unit in self.env.enemies.items():
            if not self.env.death_tracker_enemy[e_id]:
                prev_health = (
                        self.env.previous_enemy_units[e_id].health
                        + self.env.previous_enemy_units[e_id].shield
                )
                if e_unit.health == 0:
                    # delta_deaths += self.env.reward_death_value
                    delta_enemy += prev_health
                else:
                    delta_enemy += prev_health - e_unit.health - e_unit.shield

        reward = - delta_enemy + delta_ally + delta_deaths

        return reward

    def pre_train_reward_model(self):
        T = 0
        F = 0
        acc = 0
        loss_all = 0
        mean_loss = 0
        for itr in range(100000):
            reward_all, value_true = self.generate_episode(itr, evaluate=False, pre_train=True)
            loss = abs(reward_all - value_true)
            loss_all += loss
            if loss < 10 and reward_all * value_true > 0:
                T += 1
            else:
                F += 1
            if itr%1000 == 0:
                acc = T / (T + F)
                mean_loss = loss_all / 100
                loss_all = 0
                T = 0
                F = 0
                print("acc:" + str(acc) + "   loss:" + str(loss))
            # if itr % 100 == 0:
            #     acc = T / (T + F)
            #     mean_loss = loss_all/100
            #     loss_all = 0
            #     T = 0
            #     F = 0
            # print(f'\ritr: {itr}, reward_all: {reward_all}, value_true: {value_true}, loss:{loss}, acc:{acc}, mean_loss:{mean_loss}', end='', flush=True)
