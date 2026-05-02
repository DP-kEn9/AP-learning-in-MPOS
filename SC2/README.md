# SC2 Multi-Agent Reinforcement Learning

This project contains Python implementations and training scripts for multi-agent reinforcement learning experiments, including value decomposition methods and custom matrix/MMDP-style environments.

## Structure

- `train_target.py`, `train_adv.py`, `train_ori_reward_adv.py`: training entry points.
- `evaluation.py`: evaluation entry point.
- `common/`: runners, buffers, plotting helpers, and argument configuration.
- `network/`: neural network and mixer modules.
- `policy/`: policy and learner implementations.
- `envs/`: custom environments.
- `imgs/`: result images.

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

Some scripts import `smac.env.StarCraft2Env`, so the SMAC/StarCraft II environment must be installed separately when running StarCraft II experiments.
Before running the code, add the required number of task-agnostic agents to the SMAC map you want to use, save the modified map in the SMAC MAP directory, and then add or update the map agent information in the map registry of the SMAC source code.

## Usage

Run a training script from the project root, for example:

```bash
python -m torch.distributed.launch --nproc_per_node=2 train_adv.py --map='8m_1adv' --alg='qmix' --max_steps=50000000 --epsilon_anneal_steps=50000 --num=1 --gpu='0' 
```

Adjust experiment parameters in `common/arguments.py` or pass supported command-line arguments where available.

## Notes

For anonymization purposes, hard-coded path assignments in the project have been replaced with empty strings. Before running the project, configure the model, result, or data storage paths according to your local environment.
