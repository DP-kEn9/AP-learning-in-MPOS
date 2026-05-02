from smac.env import StarCraft2Env
from envs.matrix_game_2 import Matrix_game2Env
from envs.matrix_game_3 import Matrix_game3Env
from envs.uni_mmdp import uni_mmdp_Env
from envs.env_GoOrderly import EnvGoOrderly
from agents import Agents
from common.arguments import *
from adv_agents import AdvAgents
import numpy as np
import torch


def evaluate(env, env_info, agents, adv_agent, evaluate_num=100):
    win_number = 0
    episodes_reward = []
    for itr in range(evaluate_num):
        episode_reward, win = generate_episode(env, env_info, agents, adv_agent)
        episodes_reward.append(episode_reward)
        if win:
            win_number += 1
    return win_number / evaluate_num, episodes_reward


def generate_episode(env, env_info, agents, adv_agent):
    n_agents = env_info["n_agents"]
    n_actions = env_info['n_actions']
    episode_limit = env_info['episode_limit']
    env.reset()
    done = False
    info = None
    win = False
    last_action = np.zeros((n_agents, n_actions))
    epsilon = 0
    states, former_states = [], []
    state = env.get_state()
    obs = env.get_obs()
    episode_buffer = None
    avail_actions = []
    agents.policy.init_hidden(1)
    adv_agent.policy.init_hidden(1)
    for agent_id in range(n_agents):
        avail_action = env.get_avail_agent_actions(agent_id)
        avail_actions.append(avail_action)
    episode_reward = 0
    for step in range(episode_limit):
        if done:
            break
        else:
            actions, onehot_actions = [], []
            for agent_id in range(n_agents):
                if agent_id == n_agents - 1:
                    action, _ = adv_agent.choose_action(obs[0], last_action[0], 0,
                                                     avail_actions[agent_id], epsilon, True)

                else:
                    action, _ = agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                      avail_actions[agent_id], epsilon, True)

                onehot_action = np.zeros(n_actions)
                onehot_action[action] = 1
                onehot_actions.append(onehot_action)
                if type(action) == torch.Tensor:
                    action = action.cpu()

                actions.append(action)


                last_action[agent_id] = onehot_action

            reward, done, info = env.step(actions)
            if not done:
                next_obs = env.get_obs()
                next_state = env.get_state()
            else:
                next_obs = obs
                next_state = state
            next_avail_actions = []
            for agent_id in range(n_agents):
                avail_action = env.get_avail_agent_actions(agent_id)
                next_avail_actions.append(avail_action)
            episode_reward += reward
            obs = next_obs
            state = next_state
            avail_actions = next_avail_actions
    if info.__contains__('battle_won'):
        win = True if done and info['battle_won'] else False

    return episode_reward, win


def get_env(map, difficulty='7', step_mul=8, replay_dir=''):
    if map == 'matrix_2':

        return Matrix_game2Env()
    elif map == 'matrix_3':
        return Matrix_game3Env(n_agents=2,
                               n_actions=3,
                               episode_limit=1,
                               obs_last_action=False,
                               state_last_action=False,
                               print_rew=False,
                               is_print=False)
    elif 'mmdp-' in map:
        length = int(re.findall(r'\d+\.\d+|\d+', map)[-1])
        return uni_mmdp_Env(episode_limit=length)
    elif map == 'go_orderly':
        return EnvGoOrderly(map_size=6, num_agent=3)
    else:

        return StarCraft2Env(map_name=map,
                             difficulty=difficulty,
                             step_mul=step_mul,
                             replay_dir=replay_dir)


def get_info(env, arg):
    env_info = env.get_env_info()

    arg.n_actions = env_info['n_actions']

    arg.n_agents = env_info['n_agents']

    arg.state_shape = env_info['state_shape']

    arg.obs_shape = env_info['obs_shape']

    if arg.her:
        arg.obs_shape += 2

    arg.episode_limit = env_info['episode_limit']
    print(arg.obs_shape, arg.state_shape)

    if 'mmdp' in arg.map:
        arg.unit_dim = env.unit_dim
    elif not 'matrix' in arg.map and\
            not 'go_orderly' in arg.map:
        arg.unit_dim = 4 + env.shield_bits_ally + env.unit_type_bits
    return arg


if __name__ == "__main__":
    env = get_env("8m")
    env_info = env.get_env_info()
    arguments = get_common_args()
    arguments = get_q_decom_args(arguments)
    arguments = get_info(env, arguments)
    arguments.load_model = True
    arguments.cuda = False
    arguments.load_model = True
    arguments.device = 'cpu'
    arguments.load_dir = ""
    arguments.n_agents = 7
    agents = Agents(arguments)
    arguments.load_model = False
    adv_agent = AdvAgents(arguments)
    win_rate, reward = evaluate(env, env_info, agents, adv_agent)
    print(win_rate)
