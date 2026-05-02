

from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_q_decom_args
from common.ori_reward_runner import Runner
import time
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool
import os
from envs.matrix_game_2 import Matrix_game2Env
from envs.matrix_game_3 import Matrix_game3Env
from envs.mmdp_game_1 import mmdp_game1Env
from envs.uni_mmdp import uni_mmdp_Env
from envs.env_GoOrderly import EnvGoOrderly
import torch
import numpy as np
import re


def main(env, arg, itr, seed=None):
    arg.device = torch.device('cuda', arguments.local_rank)

    arg.load_model = True
    arg.load_dir = ""

    runner = Runner(env, arg, itr, seed)


    if arguments.learn:


        runner.run()

    runner.save_results()
    runner.plot()


def get_env(arg):
    if arguments.map == 'matrix_2':

        return Matrix_game2Env()
    elif arguments.map == 'matrix_3':
        return Matrix_game3Env(n_agents=2,
                               n_actions=3,
                                episode_limit=1,
                               obs_last_action=False,
                               state_last_action=False,
                               print_rew=False,
                               is_print=False)
    elif 'mmdp-' in arguments.map:
        length = int(re.findall(r'\d+\.\d+|\d+', arg.map)[-1])
        return uni_mmdp_Env(episode_limit=length)
    elif arguments.map == 'go_orderly':
        return EnvGoOrderly(map_size=6, num_agent=3)
    else:

        return StarCraft2Env(map_name=arg.map,
                             difficulty=arg.difficulty,
                             step_mul=arg.step_mul,
                             replay_dir=arg.replay_dir)


def get_info(env, arg):
    env_info = env.get_env_info()

    arg.n_actions = env_info['n_actions']

    arg.n_agents = env_info['n_agents']
    arg.n_adv = env_info['n_adv']

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


def print_error(value):
    print("error: ", value)


if __name__ == '__main__':
    start = time.time()

    start_time = datetime.utcnow().astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d_%H-%M-%S")
    print('Start time: ' + start_time)
    arguments = get_common_args()

    if 'mmdp' in arguments.map:
        arguments.min_epsilon = 1
        arguments.target_update_period = 1
        arguments.didactic = True

        arguments.max_steps = 60000
        arguments.evaluate_num = 32
        arguments.bs_rate = 1

    arguments = get_q_decom_args(arguments)

    if arguments.didactic:
        arguments.evaluation_steps_period = 500
        arguments.save_model_period = arguments.max_steps // 10

    if arguments.gpu is not None:
        arguments.cuda = True

        arguments.device = torch.device('cuda', arguments.local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = '7,8,9'




    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        arguments.cuda = False
        arguments.device = torch.device('cpu')

    environment = get_env(arguments)


    arguments = get_info(environment, arguments)



    print('min_epsilon: {}'.format(arguments.min_epsilon))
    print('Map: {}'.format(arguments.map))


    if arguments.num > 1:
        p = Pool(12)
        for i in range(arguments.num):
            if arguments.didactic:
                seed = int(time.time() % 10000000)
            else:
                seed = i
            p.apply_async(main, args=(environment, arguments, str(i) + '-' + start_time, seed), callback=print_error)
            time.sleep(5)
            print('Subprocess started...')
        p.close()
        p.join()
        print('All subprocesses finished!')
    else:
        if arguments.didactic:
            seed = None
        else:
            seed = 0
        main(environment, arguments, str(arguments.local_rank), seed)

    duration = time.time() - start
    time_list = [0, 0, 0]
    time_list[0] = duration // 3600
    time_list[1] = (duration % 3600) // 60
    time_list[2] = round(duration % 60, 2)
    print('Elapsed time: ' + str(time_list[0]) + ' hours ' + str(time_list[1]) + ' minutes ' + str(time_list[2]) + ' seconds')
    print('Start time: ' + start_time)
    end_time = datetime.utcnow().astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d_%H-%M-%S")
    print('End time: ' + end_time)
