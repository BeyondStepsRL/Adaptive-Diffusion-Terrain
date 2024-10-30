import torch
import torch.nn as nn
import torch.functional as F
from rl_agents.teacher_net import TeacherNet, transform_depth_image

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, action_noise_std, 
                 num_policy_network_layers=3, 
                 num_value_network_layers=3, 
                 policy_network_hidden_dim=256,
                 value_network_hidden_dim=256,
                 action_lower=torch.tensor([-2.0, -3.0], device=torch.device('cuda')),
                 action_upper=torch.tensor([2.0, 3.0], device=torch.device('cuda'))
                 ):
        """
        We use a value network and a policy network to compute the state-value and the actions. 
        """
        super(ActorCriticNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim
        # self.activation = nn.ELU
        self.policy_activation = nn.Tanh
        self.value_activation = nn.Tanh
        self.action_lower = action_lower
        self.action_upper = action_upper

        # policy
        if state_dim < 20:
            action_layers = [nn.Linear(self.input_dim, policy_network_hidden_dim), self.policy_activation()]
            for i in range(num_policy_network_layers):
                if i == num_policy_network_layers - 1:
                    action_layers.append(nn.Linear(policy_network_hidden_dim, action_dim))
                    # normalize the action to [-1, 1]
                    action_layers.append(nn.Tanh())
                else:
                    action_layers.append(nn.Linear(policy_network_hidden_dim, policy_network_hidden_dim))
                    action_layers.append(self.policy_activation())
        elif state_dim > 1000:
            self.teacher_net = TeacherNet()
            self.teacher_net.to(torch.device('cuda'))

        # value function
        if state_dim < 20:
            value_layers = [nn.Linear(self.input_dim, value_network_hidden_dim), self.value_activation()]
            for i in range(num_value_network_layers):
                if i == num_value_network_layers - 1:
                    value_layers.append(nn.Linear(value_network_hidden_dim, 1))
                else:
                    value_layers.append(nn.Linear(value_network_hidden_dim, value_network_hidden_dim))
                    value_layers.append(self.value_activation())

        if state_dim < 20:
            self.action_layers = nn.Sequential(*action_layers)
            self.value_layers = nn.Sequential(*value_layers)
        self.logstd = torch.nn.Parameter(torch.log(action_noise_std))

    def act(self, state, stochastic=True):
        """
        Computes the action based on the current policy.
        If stochastic is True, then the action is sampled from the distribution.
        Otherwise, it is the mean of the distribution.
        """
        if torch.isnan(state).any():
            print("NAN in state")
        
        if self.state_dim < 20:
            action_mu = self.action_layers(state)
        elif self.state_dim > 1000:
            action_mu, _ = self.teacher_net(state)
        
        # unnomralize the action to the original range
        action_mu = (self.action_upper - self.action_lower) * (action_mu + 1.0) / 2.0 + self.action_lower
        # print(action_mu)
        self.action_dist = torch.distributions.Normal(action_mu, self.logstd.exp())
        if stochastic:
            return self.action_dist.sample()
        else:
            return action_mu

    def compute_state_value(self, state):
        if torch.isnan(state).any():
            print("NAN in state")
        
        if self.state_dim < 20:
            state_value = self.value_layers(state)
        elif self.state_dim > 1000:
            _, state_value = self.teacher_net(state)

        return state_value

    def evaluate(self, state, evaluated_action=None, stochastic=False):
        """
        Computes the action, action log probability, and the state values.
        """
        if torch.isnan(state).any():
            print("NAN in state")
        if self.state_dim < 20:
            action = self.act(state, stochastic=stochastic)
            state_value = self.compute_state_value(state)
        elif self.state_dim > 1000:
            action, state_value = self.teacher_net(state)

            # num_agents = state.size(0)
            # batch_size = 25 * 100
            # if num_agents <= batch_size:
            #     action, state_value = self.teacher_net(self.elevation, state)
            # else:
            #     actions = []
            #     state_values = []
            #     for i in range(0, num_agents, batch_size):
            #         state_batch = state[i:i+batch_size]
            #         elevation_batch = self.elevation[i:i+batch_size]

            #         action_batch, state_value_batch = self.teacher_net(elevation_batch, state_batch)

            #         actions.append(action_batch)
            #         state_values.append(state_value_batch)

            #     action = torch.cat(actions, dim=0)
            #     state_value = torch.cat(state_values, dim=0)

            # unnomralize the action to the original range
            action = (self.action_upper - self.action_lower) * (action + 1.0) / 2.0 + self.action_lower
            self.action_dist = torch.distributions.Normal(action, self.logstd.exp())
            if stochastic:
                action = self.action_dist.sample()

        if evaluated_action != None:
            action_log_prob = self.action_dist.log_prob(evaluated_action)
        else:
            action_log_prob = self.action_dist.log_prob(action)
        return action, action_log_prob, state_value

    def compute_entropy(self, state):
        """
        Computes the entropy of the policy at the given state.
        """
        return self.action_dist.entropy().sum(-1)
    
    def compute_encoder_weight_decay(self):
        """
        Compute the weight decay for the encoder network.
        """
        if self.privileged_obs_dim > 0:
            reg = 0.0
            for layer in self.encoder_layers:
                for param in layer.parameters():
                    reg += (param ** 2).sum() 
            return reg * 1e-5
        else:
            return 0.0