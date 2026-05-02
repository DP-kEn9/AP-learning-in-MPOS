from common.arguments import get_common_args, get_q_decom_args
import time
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool
import os
from common.target_runner import Runner
import torch
import numpy as np
import re
import gymnasium
import highway_env


def main(env, arg, itr, seed=None):
                                                             
    arguments.device = 'cpu'
    arg.load_model = False
    arg.load_dir = ""
    arg.load_dir = ""
    runner = Runner(env, arg, itr, seed)
                                      
            
    if arguments.learn:
        if arg.reward_model == 1:
            runner.pre_train_reward_model()
        runner.run()
                            
    runner.save_results()
    runner.plot()


def get_env(arg):
    env = gymnasium.make(
        arg.map,
        render_mode="rgb_array",
        config={
            "controlled_vehicles": arg.control_units,                           
            "vehicles_count": arg.other_vehicle,                                                         
        }
    )
    env.reset(seed=0)
    arg.obs_type = "TimeToCollision"
    if arg.map == "highway-v0":
        arg.obs_type = "Kinematics"
    action_type = "DiscreteMetaAction"
    if arg.map == "racetrack-v0":
        action_type = "ContinuousAction"
    env.unwrapped.config.update({
        "controlled_vehicles": arg.control_units,
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": arg.obs_type,
            }
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {
                "type": action_type,
            }
        }
    })
    return env


def print_error(value):
    print("error: ", value)


def get_info(env, arg):
    if arg.training == "adv":
        arg.agent_pos = np.arange(arg.control_units)[-arg.n_agents:]
    else:
        arg.agent_pos = np.arange(arg.control_units)[:arg.n_agents]
          
              
    env.reset()
    arg.n_actions = int(env.action_space.spaces[0].n)
    x, y = env.observation_space.spaces[0].shape[0], env.observation_space.spaces[0].shape[1]
              
    arg.obs_shape = x*y
    if arg.obs_type == "TimeToCollision":
        z = env.observation_space.spaces[0].shape[2]
        arg.obs_shape = x*y*z
    arg.state_shape = arg.obs_shape * arg.n_agents
                 
    if arg.her:
        arg.obs_shape += 2
                 
    arg.episode_limit = 200
    print(arg.obs_shape, arg.state_shape)
            
                           
                                     
                                        
                                          
                                                                      
    return arg


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
                                                     
        arguments.device = torch.device('cuda', arguments.gpu)
                                  
                  
               
                                                                
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        arguments.cuda = False
        arguments.device = torch.device('cpu')

    environment = get_env(arguments)

            
    arguments = get_info(environment, arguments)
                                                               
                            

    print('min_epsilon: {}'.format(arguments.min_epsilon))
    print('Map: {}'.format(arguments.map))

                                       
    if arguments.num > 1:
        p = ""
        for i in range(arguments.num):
            if arguments.didactic:
                seed = int(time.time() % 10000000)
            else:
                seed = i
            p.apply_async(main, args=(environment, arguments, str(i) + '-' + start_time, seed), callback=print_error)
            time.sleep(5)
        print('Subprocesses started...')
        p.close()
        p.join()
        print('All subprocesses finished.')
    else:
        if arguments.didactic:
            seed = None
        else:
            seed = 0
        main(environment, arguments, start_time, seed)

    duration = time.time() - start
    time_list = [0, 0, 0]
    time_list[0] = duration // 3600
    time_list[1] = (duration % 3600) // 60
    time_list[2] = round(duration % 60, 2)
    print('Elapsed time: ' + str(time_list[0]) + ' h ' + str(time_list[1]) + ' min ' + str(time_list[2]) + ' s')
    print('Start time: ' + start_time)
    end_time = datetime.utcnow().astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d_%H-%M-%S")
    print('End time: ' + end_time)
