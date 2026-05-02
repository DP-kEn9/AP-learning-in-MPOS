# Highway Multi-Agent RL

This project provides experimental code for multi-agent reinforcement learning based on `highway-env` and `gymnasium`. It includes the training entry point, replay buffer, environment interaction runner, neural network modules, and implementations of several learning policies.

## Project Structure

- `train.py`: Training entry point for environment construction, argument loading, and training execution.
- `agents.py`: Agent action selection and training orchestration.
- `common/`: Shared arguments, runner logic, replay buffer, plotting utilities, and model conversion tools.
- `network/`: Network architectures, including RNN, QMIX mixer, and reward networks.
- `policy/`: Policy and learner implementations, including QMIX, DMAQ, and PPO.
- `test.py`: Basic test script.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m torch.distributed.launch --nproc_per_node=2 train_adv.py
```

Configuration options such as the map, number of agents, training steps, algorithm, and GPU settings can be adjusted through command-line arguments. See `arguments.py` for details. For example:

```bash
python -m torch.distributed.launch --nproc_per_node=2 train_adv.py --alg='qmix' --max_steps=50000000 --epsilon_anneal_steps=50000 --num=1 --gpu='0'
```

## Notes

For anonymization purposes, hard-coded path assignments in the project have been replaced with empty strings. Before running the project, configure the model, result, or data storage paths according to your local environment.
