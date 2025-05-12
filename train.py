import argparse
import os
import random
import numpy as np
import torch
import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
import DQN_network
import DQN_memory
import DQN_policy
import DQN_preprocessors
import dqn

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_output_folder(parent_dir, env_name, exp_id=None):
    os.makedirs(parent_dir, exist_ok=True)
    if exp_id is None:
        experiment_id = 0
        for folder_name in os.listdir(parent_dir):
            try:
                folder_num = int(folder_name.split('_')[0])
                experiment_id = max(experiment_id, folder_num)
            except ValueError:
                continue
        experiment_id += 1
    else:
        experiment_id = exp_id
    path = os.path.join(parent_dir, f'{experiment_id:03d}_{env_name}')
    os.makedirs(path, exist_ok=True)
    return path, experiment_id

def main():
    parser = argparse.ArgumentParser(description="Train DQN on Atari with PyTorch")
    parser.add_argument('--env', default='SpaceInvadersNoFrameskip-v4')
    parser.add_argument('--output', default='atari-results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--optimizer', default='rmsprop')
    parser.add_argument('--learning_rate', type=float, default=0.00025)
    parser.add_argument('--model', default='convnet')
    parser.add_argument('--max_iters', type=int, default=12500000)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--render_path', type=str, default='/dev/null/')
    parser.add_argument('--std_img', action='store_true')
    parser.add_argument('--train_policy', default='anneal', choices=['anneal', 'epgreedy'])
    parser.add_argument('--exp_id', type=int, default=None)
    parser.add_argument('--target_update_freq', type=int, default=10000)
    parser.add_argument('--train_freq', type=int, default=4)
    parser.add_argument('--mem_size', type=int, default=100000)
    parser.add_argument('--learning_type', default='normal', choices=['normal', 'double'])
    parser.add_argument('--final_eval', action='store_true')
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    args.output, experiment_id = get_output_folder(args.output, args.env, args.exp_id)

    env = gym.make(args.env,render_mode="rgb_array",frameskip=4)
    # env = AtariPreprocessing(env, terminal_on_life_loss=True) #for Atari processing?
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    num_actions = env.action_space.n

    # Build Q-networks
    input_size = 84
    frame_history = 4
    input_shape = (frame_history, input_size, input_size)  # Assuming 4 stacked grayscale frames
    q_network = DQN_network.get_model(args.model,input_shape, num_actions).to(device)
    target_network = DQN_network.get_model(args.model,input_shape, num_actions).to(device)
    # target_network.load_state_dict(q_network.state_dict())

    # Policy
    if args.train_policy == 'anneal':
        policy = DQN_policy.LinearDecayGreedyEpsilonPolicy(start_value=1.0, end_value=0.1, num_steps=100000)
    else:
        policy = DQN_policy.GreedyEpsilonPolicy(epsilon=0.1)

    # Optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(q_network.parameters(), lr=args.learning_rate, alpha=0.95, eps=0.01)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(q_network.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")

    #preprocessor
    preproc = DQN_preprocessors.PreprocessorSequence(
      [DQN_preprocessors.AtariPreprocessor(input_size, args.std_img),
       DQN_preprocessors.HistoryPreprocessor(frame_history)])
    
    learning_type = dqn._LEARNING_TYPE_NORMAL
    if args.learning_type == 'double':
        learning_type = dqn._LEARNING_TYPE_DOUBLE
    # Replay Buffer
    memory = DQN_memory.BasicReplayMemory(args.mem_size)

    agent = dqn.DQNAgent(
        q_network=q_network,#
        target_q_network=target_network,#
        optimizer=optimizer,#
        memory=memory,#
        policy=policy,#
        gamma=0.99,#
        batch_size=32,#
        target_update_freq=args.target_update_freq,#
        train_freq=args.train_freq,#
        checkpoint_dir=args.output,#
        experiment_id=experiment_id,#
        learning_type=learning_type,#
        loss_func=torch.nn.MSELoss(),#
        num_burn_in=20000,#
        preprocessor=preproc,#
        env_name=args.env
    )

    if args.checkpoint:
        agent.load(args.checkpoint)

    if args.final_eval:
        agent.evaluate(env, num_episodes=100, max_episode_length=1e4, render=False)
    elif args.render:
        agent.evaluate(env, num_episodes=1, max_episode_length=1e4, render=True)
    else:
        agent.fit(env, args.max_iters)

    env.close()

if __name__ == '__main__':
    main()
