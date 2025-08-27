from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_q_decom_args
from common.adv_runner import Runner
import time
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool
import os
import torch
import numpy as np
import re


def main(env, arg, itr, seed=None):
    # arg.device = torch.device('cuda', arguments.local_rank)
    arguments.device = 'cpu'
    arg.load_model = True
    arg.load_dir = ""
    runner = Runner(env, arg, itr, seed)
    # print(runner.get_initial_qtot())
    if arguments.learn:
        # if arg.reward_model == 1:
        #     runner.pre_train_reward_model()
        runner.run()
        # runner.run_steps()
    runner.save_results()
    runner.plot()


def get_env(arg):
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
    elif not 'matrix' in arg.map and \
            not 'go_orderly' in arg.map:
        arg.unit_dim = 4 + env.shield_bits_ally + env.unit_type_bits
    return arg


def print_error(value):
    print("error: ", value)


if __name__ == '__main__':
    start = time.time()
    start_time = datetime.utcnow().astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d_%H-%M-%S")
    arguments = get_common_args()

    if 'mmdp' in arguments.map:
        arguments.min_epsilon = 1
        arguments.target_update_period = 1
        arguments.didactic = True
        # 600
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
    # arguments.n_agents = arguments.n_agents - arguments.n_adv
    # arguments.tar_agents = 7


    if arguments.num > 1:
        p = Pool(12)
        for i in range(arguments.num):
            if arguments.didactic:
                seed = int(time.time() % 10000000)
            else:
                seed = i
            p.apply_async(main, args=(environment, arguments, str(i) + '-' + start_time, seed), callback=print_error)
            time.sleep(5)
        p.close()
        p.join()
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
    end_time = datetime.utcnow().astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d_%H-%M-%S")
