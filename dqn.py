import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


# Deep Q-Network
class DQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_actions)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions


# DQN Agent
class DQNAgent:
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dims,
                 epsilon_dec=0.996, epsilon_end=0.01, mem_size=10000, replace_target=1000, tau=10, expert=False):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_end
        self.epsilon_dec = epsilon_dec
        self.lr = alpha
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.replace_target = replace_target
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.tau = tau
        self.expert = expert


        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.expert_memory = ReplayBuffer(mem_size, input_dims, n_actions)
        

        self.q_eval = DQNetwork(input_dims, n_actions)
        self.q_next = DQNetwork(input_dims, n_actions)

        self.optimizer = optim.Adam(self.q_eval.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        self.q_next.load_state_dict(self.q_eval.state_dict())
        self.q_next.eval()

    def remember(self, state, action, reward, new_state, done):
        if self.expert:
            self.expert_memory.store_transition(state, action, reward, new_state, done)
        else:
            self.memory.store_transition(state, action, reward, new_state, done)
        

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        q_values = self.q_eval(state)

        # Debugging outputs
        # print("State:", state)
        # print("Q-values:", q_values)

        if torch.any(torch.isnan(q_values)) or torch.any(torch.isinf(q_values)):
            q_values = torch.zeros_like(q_values)

        # Normalize Q-values to prevent overflow
        scaled_qs = q_values - torch.max(q_values)
        exp_qs = torch.exp(scaled_qs / self.tau)
        sum_exp_qs = torch.sum(exp_qs)

        # Compute probabilities
        if sum_exp_qs == 0 or torch.any(torch.isnan(sum_exp_qs)):
            # Sum of exp(Q-values) is zero or NaN.
            probs = torch.ones_like(exp_qs) / len(exp_qs)  # Uniform distribution
        else:
            probs = exp_qs / (sum_exp_qs + 1e-9)

        action = np.random.choice(self.action_space, p=probs.detach().numpy())
        return action
    
    
    def sample_buffer(self, batch_size, expert_new_ratio=0.5):
        expert_num = min(self.expert_memory.mem_cntr, int(expert_new_ratio * batch_size))
        new_num = min(self.memory.mem_cntr, batch_size - expert_num)
        
        if expert_num + new_num < batch_size:
            print("Not enough samples in buffers.")
            return None

        expert = self.expert_memory.sample_buffer(expert_num)
        new = self.memory.sample_buffer(new_num)

        states = np.concatenate((expert[0], new[0]))
        actions = np.concatenate((expert[1], new[1]))
        rewards = np.concatenate((expert[2], new[2]))
        states_ = np.concatenate((expert[3], new[3]))
        terminal = np.concatenate((expert[4], new[4]))

        return states, actions, rewards, states_, terminal

    def learn(self):
        if self.memory.mem_cntr + self.expert_memory.mem_cntr < self.batch_size:
            return

        self.optimizer.zero_grad()

        # Sample from buffers
        buffer_data = self.sample_buffer(self.batch_size)
        if buffer_data is None:
            return

        states, actions, rewards, states_, terminal = buffer_data

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        states_ = torch.tensor(states_, dtype=torch.float32)
        terminal = torch.tensor(terminal, dtype=torch.float32)

        # Compute Q-values and target
        indices = np.arange(self.batch_size)
        q_pred = self.q_eval(states)[indices, actions]
        q_next = self.q_next(states_).max(dim=1)[0].detach()
        q_target = rewards + self.gamma * q_next * terminal

        # Compute loss and backprop
        loss = self.loss(q_pred, q_target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.q_eval.parameters(), max_norm=10)
        self.optimizer.step()

        self.learn_step_counter += 1

        # Replace target network
        if self.learn_step_counter % self.replace_target == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
            
        self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)




    def save_model(self, filename):
        checkpoint = {
            'q_eval_state_dict': self.q_eval.state_dict(),
            'q_next_state_dict': self.q_next.state_dict(),
            'replay_buffer': {
                'state_memory': self.memory.state_memory,
                'new_state_memory': self.memory.new_state_memory,
                'action_memory': self.memory.action_memory,
                'reward_memory': self.memory.reward_memory,
                'terminal_memory': self.memory.terminal_memory,
                'mem_cntr': self.memory.mem_cntr
            },
            'expert_buffer': {
                'state_memory': self.expert_memory.state_memory,
                'new_state_memory': self.expert_memory.new_state_memory,
                'action_memory': self.expert_memory.action_memory,
                'reward_memory': self.expert_memory.reward_memory,
                'terminal_memory': self.expert_memory.terminal_memory,
                'mem_cntr': self.expert_memory.mem_cntr
            },
            'epsilon': self.epsilon,
            'learn_step_counter': self.learn_step_counter
        }
        torch.save(checkpoint, filename)


    def load_model(self, filename):
        checkpoint = torch.load(filename)
        
        # Load Q-network states
        if not self.expert:
            self.q_eval.load_state_dict(checkpoint['q_eval_state_dict'])
            self.q_next.load_state_dict(checkpoint['q_next_state_dict'])

        # Load ReplayBuffer
        replay_buffer_data = checkpoint['replay_buffer']
        self.memory.state_memory = replay_buffer_data['state_memory']
        self.memory.new_state_memory = replay_buffer_data['new_state_memory']
        self.memory.action_memory = replay_buffer_data['action_memory']
        self.memory.reward_memory = replay_buffer_data['reward_memory']
        self.memory.terminal_memory = replay_buffer_data['terminal_memory']
        self.memory.mem_cntr = replay_buffer_data['mem_cntr']
        
        # Load Expert memory
        replay_buffer_data = checkpoint['expert_buffer']
        self.expert_memory.state_memory = replay_buffer_data['state_memory']
        self.expert_memory.new_state_memory = replay_buffer_data['new_state_memory']
        self.expert_memory.action_memory = replay_buffer_data['action_memory']
        self.expert_memory.reward_memory = replay_buffer_data['reward_memory']
        self.expert_memory.terminal_memory = replay_buffer_data['terminal_memory']
        self.expert_memory.mem_cntr = replay_buffer_data['mem_cntr']

        # Load other parameters
        self.epsilon = checkpoint['epsilon']
        self.learn_step_counter = checkpoint['learn_step_counter']

