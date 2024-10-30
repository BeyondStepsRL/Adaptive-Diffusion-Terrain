import torch

class RolloutBuffer:
    def __init__(self, 
                 num_envs, 
                 num_transitions, 
                 state_dim, 
                 action_dim, 
                 gamma=0.99, 
                 lam=0.95) -> None:
        """
        Stores the agent's most recent N transitions, which are used for on-policy training. 
        If lam=1, we use Monte Carlo return: G_t = r_t + gamma*r_{t+1} + ... + gamma^{T}v(s_T).
        If lam=0, we use TD(0) return: G_t = r_t + gamma*v(s_{t+1}).
        The data contain: 
            (1) state: the current state
            (2) action: the action taken
            (3) next_state: next observed state
            (4) reward: reward of the transition
            (5) done: whether the current transition is terminal
            (6) action_logprob: the log probability of the action taken 
            (7) value: v(s), the current state value
            (8) return: G(t), the target value for the value function
        """
        self.device = torch.device('cuda')
        self.state = torch.empty((num_transitions, num_envs, state_dim), device=self.device) 
        self.actions = torch.empty((num_transitions, num_envs, action_dim), device=self.device)
        self.old_action_mu = torch.empty((num_transitions, num_envs, action_dim), device=self.device)
        self.old_action_sigma = torch.empty((num_transitions, num_envs, action_dim), device=self.device)
        self.next_state = torch.empty((num_transitions, num_envs, state_dim), device=self.device)
        self.reward = torch.empty((num_transitions, num_envs, 1), device=self.device)
        self.dones = torch.empty((num_transitions, num_envs, 1), device=self.device)
        self.action_logprob = torch.empty((num_transitions, num_envs, action_dim), device=self.device)
        self.value = torch.empty((num_transitions, num_envs, 1), device=self.device)
        self.returns = torch.empty((num_transitions, num_envs, 1), device=self.device)

        self.num_envs = num_envs
        self.num_transitions = num_transitions
        self.num_samples_per_batch = num_envs * num_transitions
        self.step = 0
        self.gamma = gamma
        self.lam = lam
        self.action_dim = action_dim
        self.state_dim = state_dim

    def compute_return_and_advantage(self, last_values):
        """
        Computes the return and the advantage function for each state-action pair in the buffer.
        The return is served as the target for the value function, and the advantage function is
        used for the policy gradient.
        """
        advantage = 0
        for step in reversed(range(self.num_transitions)):
            if step == self.num_transitions - 1:
                next_values = last_values
            else:
                next_values = self.value[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.reward[step] + next_is_not_terminal * self.gamma * next_values - self.value[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            self.returns[step] = advantage + self.value[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.value
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def clear(self):
        self.step = 0

    def full(self):
        """
        Check whether the buffer is full
        """
        return self.step == self.num_transitions
    
    def sample_batch(self, num_epochs, mini_batch_size):
        # generate random indices
        num_mini_batches = self.num_samples_per_batch // mini_batch_size 
        indices = torch.randperm(self.num_samples_per_batch, device=self.device)
        # flatten the transitions 
        state = self.state.view(-1, self.state_dim)
        action = self.actions.view(-1, self.action_dim)
        old_action_mu = self.old_action_mu.view(-1, self.action_dim)
        old_action_sigma = self.old_action_sigma.view(-1, self.action_dim)
        value_targets = self.returns.view(-1, 1)
        old_action_logprobs = self.action_logprob.view(-1, self.action_dim)
        advantage_fn = self.advantages.view(-1, 1)
        for _ in range(num_epochs):
            for mini_batch_id in range(num_mini_batches):
                start_id = mini_batch_id * mini_batch_size 
                end_id = (mini_batch_id + 1) * mini_batch_size 
                selected_indices = indices[start_id:end_id]
                yield state[selected_indices], \
                      action[selected_indices], \
                      old_action_mu[selected_indices], \
                      old_action_sigma[selected_indices], \
                      value_targets[selected_indices], \
                      old_action_logprobs[selected_indices], \
                      advantage_fn[selected_indices]

    def add_sample(self, 
                   state, 
                   action, 
                   old_action_mu, 
                   old_action_sigma, 
                   next_state, 
                   reward, 
                   dones, 
                   action_logprob, 
                   value):
        """
        Each element has dimension of (num_envs, *).
        """
        self.state[self.step] = state.detach().clone().view(self.num_envs, -1)
        self.actions[self.step] = action.detach().clone().view(self.num_envs, -1)
        self.old_action_mu[self.step] = old_action_mu.detach().clone().view(self.num_envs, -1)
        self.old_action_sigma[self.step] = old_action_sigma.detach().clone().view(self.num_envs, -1)
        self.next_state[self.step] = next_state.detach().clone().view(self.num_envs, -1)
        self.reward[self.step] = reward.detach().clone().view(self.num_envs, -1)
        self.dones[self.step] = dones.detach().clone().view(self.num_envs, -1)
        self.action_logprob[self.step] = action_logprob.detach().clone().view(self.num_envs, -1)
        self.value[self.step] = value.detach().clone().view(self.num_envs, -1)
        self.step += 1