import argparse
import re
import math


def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='8m_1adv')
    parser.add_argument('--alg', type=str, default='qmix')
    parser.add_argument('--adv_alg', type=str, default='qmix')
    parser.add_argument('--her', type=int, default=None)
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--target_update_period', type=int, default=200)
    parser.add_argument('--max_steps', type=int, default=2000000)
    parser.add_argument('--bs_rate', type=float, default=12.5)
    parser.add_argument('--optim', type=str, default='RMS')
    parser.add_argument('--result_dir', type=str, default='./results-steps')
    parser.add_argument('--model_dir', type=str, default='./model-steps')
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--learn', type=bool, default=True)
    parser.add_argument('--epsilon_anneal_steps', type=int, default=50000)
    parser.add_argument('--min_epsilon', type=float, default=0.05)
    parser.add_argument('--reward_model', type=int, default=1)
    parser.add_argument("--local-rank", type=int)

    args = parser.parse_args()

    args.difficulty = '7'
    args.step_mul = 8
    args.C_UPDATE_STEPS = 10
    args.A_UPDATE_STEPS = 10
    args.seed = 123
    args.replay_dir = ''
    args.last_action = True
    args.reuse_network = True
    args.gamma = 0.99
    args.evaluate_num = 50
    args.didactic = False
    return args


def get_q_decom_args(args):
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    # qtran "qtran_base"
    if 'qtran' in args.alg:
        args.qtran_hidden_dim = 64
        args.qtran_arch = "qtran_paper"
        args.mixing_embed_dim = 64
        args.opt_loss = 1
        args.nopt_min_loss = 0.1
        args.network_size = 'small'
        # QTRAN lambda
        args.lambda_opt = 1
        args.lambda_nopt = 1
    # qatten
    elif args.alg == 'qatten':
        args.mixing_embed_dim = 32
        args.n_head = 4  # attention head number
        args.attend_reg_coef = 0.001  # attention regulation coefficient  # For MMM2 and 3s5z_vs_3s6z, it is 0.001
        args.hypernet_layers = 2
        args.nonlinear = False  # non-linearity, for MMM2, it is True
        args.weighted_head = False  # weighted head Q-values, for MMM2 and 3s5z_vs_3s6z, it is True
        args.state_bias = True
        args.hypernet_embed = 64
        args.mask_dead = False
    elif 'dmaq' in args.alg:
        args.mixing_embed_dim = 32
        args.hypernet_embed = 64
        args.adv_hypernet_layers = 3
        args.adv_hypernet_embed = 64

        args.num_kernel = 10
        args.is_minus_one = True
        args.weighted_head = True
        args.is_adv_attention = True
        args.is_stop_gradient = True
    elif 'wqmix' in args.alg:
        args.central_loss = 1
        args.qmix_loss = 1
        args.w = 0.1  # $\alpha$ in the paper
        # False -> CW-QMIX, True -> OW-QMIX
        args.hysteretic_qmix = True
        if args.alg == 'cwqmix':
            args.hysteretic_qmix = False

        args.central_mixing_embed_dim = 256
        args.central_action_embed = 1
        args.central_mac = "basic_central_mac"
        args.central_agent = "central_rnn"
        args.central_rnn_hidden_dim = 64
        args.central_mixer = "ff"

    args.lr = 5e-4
    args.A_LR = 1e-4
    args.C_LR = 3e-4
    # epsilon greedy
    args.epsilon = 1
    args.epsilon_decay = (args.epsilon - args.min_epsilon) / args.epsilon_anneal_steps
    args.epsilon_anneal_scale = 'step'
    args.n_episodes = 1
    args.train_steps = 1
    # args.evaluation_period = 100
    # args.evaluation_steps_period = math.ceil(args.max_steps / 300.)
    args.evaluation_steps_period = 50000
    # args.batch_size = 128
    args.batch_size = 8
    # args.buffer_size = int(2 ** 17)
    args.buffer_size = int(5000)
    # args.save_model_period = 5000
    # args.save_model_period = math.ceil(args.n_itr / 200.)
    args.save_model_period = math.ceil(args.max_steps / 100.)
    args.clip_norm = 10
    return args
