import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
import copy
import gymnasium as gym

import DQN_policy
import DQN_preprocessors

PRINT_AFTER_ITER = 10000
EVAL_AFTER_ITER = 5e6
EVAL_MAX_EPISODE_LEN = 1e4
NUM_TEST_EPISODES = 20
SAVE_AFTER_ITER = 5e6
NO_OP_STEPS = 30

_LEARNING_TYPE_NORMAL = 0
_LEARNING_TYPE_DOUBLE = 1

class DQNAgent:
    def __init__(self,
                 q_network,#
                 target_q_network,#
                 preprocessor,#
                 memory,#
                 policy,#
                 gamma,#
                 target_update_freq,#
                 num_burn_in,#
                 train_freq,#
                 batch_size,#
                 optimizer,#
                 loss_func,#
                 checkpoint_dir,#
                 experiment_id,#
                 env_name,#
                 learning_type=_LEARNING_TYPE_NORMAL):
        self.env_name = env_name
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.preprocessor = preprocessor
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.experiment_id = experiment_id
        self.training_reward_seen = 0
        self.learning_type = learning_type
        self.device = next(self.q_network.parameters()).device

    def calc_q_values(self, state, preproc=None, network=None):
        if preproc is None:
            preproc = self.preprocessor
        if network is None:
            network = self.q_network
        state = torch.tensor(np.expand_dims(
            preproc.process_state_for_network(state), axis=0), dtype=torch.float32).to(self.device)
        with torch.no_grad():#why no grad?
            return network(state).cpu().numpy()

    def update_policy(self, itr):
        samples = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.tensor(self.preprocessor.process_state_for_network(np.array(states)), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(self.preprocessor.process_state_for_network(np.array(next_states)), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1).to(self.device)

        q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            if self.learning_type == _LEARNING_TYPE_DOUBLE:
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_q_network(next_states).gather(1, next_actions)
            else:
                next_q_values = self.target_q_network(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q_values * (~dones)

        loss = self.loss_func(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_reward_seen += rewards.sum().item()

        if itr % PRINT_AFTER_ITER == 0 and itr >= PRINT_AFTER_ITER:
            print(f"Iteration {itr}: Loss {loss.item():.6f}, Reward seen {self.training_reward_seen}")

        return loss.item()

    def fit(self, env, num_iterations, max_episode_length=None):
        self.preprocessor.reset()
        obs, _ = env.reset()
        obs = self.preprocessor.process_state_for_memory(obs)

        testing_env = copy.deepcopy(env)
        loss = 0
        
        # Burn-in replay buffer with random policy
        testing_env = copy.deepcopy(env)
        value_fn = np.random.random((env.action_space.n,))
        random_policy = DQN_policy.UniformRandomPolicy(env.action_space.n)
        for _ in range(self.num_burn_in):
            obs = self.push_replay_memory(obs, env, random_policy, is_training=False, value_fn=value_fn)
        
        start_time = time.time()
        
        for itr in range(num_iterations):
            value_fn=self.calc_q_values(obs, network=self.q_network)
            obs = self.push_replay_memory(obs, env, self.policy, is_training=True, value_fn=value_fn)

            if itr % self.target_update_freq == 0:
                self.target_q_network.load_state_dict(self.q_network.state_dict())

            if itr % self.train_freq == 0:
                loss = self.update_policy(itr)

            if itr % PRINT_AFTER_ITER == 0:
                print(f"Iteration {itr}: Loss {loss:.6f}, "
                    f"Speed: {PRINT_AFTER_ITER / (time.time() - start_time):.2f} it/s, "
                    f"Total reward: {self.training_reward_seen}")
                start_time = time.time()

            if itr % EVAL_AFTER_ITER == 0 and itr >= EVAL_AFTER_ITER:
                self.evaluate(testing_env, NUM_TEST_EPISODES,
                            max_episode_length=EVAL_MAX_EPISODE_LEN)

            if itr % SAVE_AFTER_ITER == 0 and itr >= SAVE_AFTER_ITER:
                self.save(itr)


    def push_replay_memory(self, state, env, policy, is_training, value_fn):
        action = policy.select_action(q_values=value_fn, is_training=is_training)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward = self.preprocessor.process_reward(reward)
        next_state_proc = self.preprocessor.process_state_for_memory(next_state)
        self.memory.append(state, action, reward, next_state_proc, done)

        if done:
            self.preprocessor.reset()
            next_state, _ = env.reset()
            next_state_proc = self.preprocessor.process_state_for_memory(next_state)

        return next_state_proc

    def save(self, itr):
        # Sanitize env_name by replacing path separators
        safe_env_name = self.env_name.replace('/', '_')
        filename = f"{self.checkpoint_dir}/{safe_env_name}_run{self.experiment_id}_iter{itr}.pt"
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.q_network.state_dict(), filename)

    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename))
    
    def evaluate(self, env, num_episodes, max_episode_length=10000, render=False):
        preproc = DQN_preprocessors.PreprocessorSequence(
          [DQN_preprocessors.AtariPreprocessor(
            self.q_network.input_shape[1],
            self.preprocessor.preprocessors[0].std_img),
           DQN_preprocessors.HistoryPreprocessor(self.q_network.input_shape[0])])
        pol = DQN_policy.GreedyEpsilonPolicy(0.05)
        self.q_network.eval()
        total_rewards = []
        for ep in range(num_episodes):
            obs, _ = env.reset()
            preproc.reset()
            done = False
            ep_reward = 0
            step = 0
            while not done and step < max_episode_length:
                obs = preproc.process_state_for_memory(obs)
                q_values = self.calc_q_values(obs, preproc=preproc)
                action = pol.select_action(q_values=q_values)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                step += 1
                if render:
                    env.render()
            print(f"Episode {ep} reward: {ep_reward}")
            total_rewards.append(ep_reward)
        avg_reward = np.mean(total_rewards)
        print(f"Average reward over {num_episodes} episodes: {avg_reward}")
        self.q_network.train()

    def run_no_op_steps(self, env):
        for _ in range(NO_OP_STEPS - 1):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                env.reset()
        obs, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            obs, _ = env.reset()
        return obs
